package com.phonecam.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.view.OrientationEventListener;
import android.view.Surface;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Service au premier plan : caméra et micro continuent quand on change d'application ou
 * qu'on éteint l'écran (notification permanente, comme une appli d'appel vidéo).
 * Il se reconnecte tout seul au PC si la liaison tombe.
 */
public class StreamService extends Service implements VideoSource.Sink, AudioSource.Sink, WsClient.Listener {
    static final String ACTION_STOP = "com.phonecam.app.STOP";
    private static final String CHANNEL = "phonecam";
    private static final long MAX_QUEUED = 400_000;   // octets en attente : au-delà, on saute des images

    /** Onglets Navigation / Appel : ce que le PC renvoie (images, état de l'appel). */
    interface WebListener {
        void onWebFrame(int kind, byte[] data);

        void onWebText(String json);
    }

    static volatile StreamService instance;
    static volatile WebListener web;
    private volatile String webMode, webView;     // renvoyés au PC après chaque reconnexion
    private final java.util.concurrent.ExecutorService textOut = java.util.concurrent.Executors.newSingleThreadExecutor();
    volatile String status = "Démarrage…";
    volatile int sentFrames, droppedFrames;

    private Prefs prefs;
    private volatile VideoSource video;
    private AudioSource audio;
    private volatile WsClient ws;
    private volatile boolean running;
    private volatile boolean needKey = true;
    private final LinkedBlockingQueue<byte[]> queue = new LinkedBlockingQueue<>();
    private final AtomicLong queued = new AtomicLong();
    private Thread netThread, sendThread;
    private PowerManager.WakeLock wakeLock;
    private WifiManager.WifiLock wifiLock;
    private OrientationEventListener orientation;
    private volatile int rotation = 0, deviceAngle = 0;
    private final Object cameraLock = new Object();
    private Surface pendingPreview;

    static void start(Context c) {
        Intent i = new Intent(c, StreamService.class);
        if (Build.VERSION.SDK_INT >= 26) c.startForegroundService(i);
        else c.startService(i);
    }

    static void stop(Context c) {
        c.stopService(new Intent(c, StreamService.class));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (running) return START_NOT_STICKY;
        prefs = Prefs.load(this);
        int types = 0;
        if (Build.VERSION.SDK_INT >= 30) {
            if (prefs.camera) types |= ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA;
            if (prefs.mic) types |= ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE;
        }
        try {
            if (Build.VERSION.SDK_INT >= 30 && types != 0) startForeground(1, notification(), types);
            else startForeground(1, notification());
        } catch (Exception e) {   // autorisation caméra/micro manquante
            status = "Impossible de démarrer : " + e.getMessage();
            stopSelf();
            return START_NOT_STICKY;
        }
        instance = this;
        running = true;
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "phonecam:flux");
        wakeLock.acquire();
        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(WIFI_SERVICE);
        // Wi-Fi sans mise en veille seulement s'il sert (pas par câble) et hors mode éco : c'est
        // l'une des grosses sources de chaleur.
        if (prefs.tls && !prefs.eco) {
            wifiLock = wm.createWifiLock(Build.VERSION.SDK_INT >= 29 ? WifiManager.WIFI_MODE_FULL_LOW_LATENCY
                    : WifiManager.WIFI_MODE_FULL_HIGH_PERF, "phonecam:wifi");
            wifiLock.acquire();
        }
        if (Build.VERSION.SDK_INT >= 29) {
            thermal = status -> {
                thermalStatus = status;
                updateEco();
            };
            pm.addThermalStatusListener(thermal);
            thermalStatus = pm.getCurrentThermalStatus();
        }
        // Ouvrir la caméra peut prendre une seconde : pas sur le fil de l'interface.
        netThread = new Thread(() -> {
            startSources();
            updateEco();
            netLoop();
        }, "connexion");
        netThread.start();
        sendThread = new Thread(this::sendLoop, "envoi");
        sendThread.setPriority(Thread.MAX_PRIORITY);
        sendThread.start();
        return START_NOT_STICKY;
    }

    private PowerManager.OnThermalStatusChangedListener thermal;
    private volatile int thermalStatus, ecoLevel = -1;

    /** Niveau d'économie : réglage « Éco » de l'appli, renforcé si le téléphone chauffe. */
    private void updateEco() {
        int level = prefs.eco ? 1 : 0;
        if (thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) level = Math.max(level, 1);
        if (thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) level = 2;
        if (level == ecoLevel) return;
        ecoLevel = level;
        VideoSource v = video;
        if (v != null) v.setEco(level);
        sendWeb("{\"t\":\"eco\",\"level\":" + level + "}");   // le PC envoie aussi moins d'images
    }

    int ecoLevel() {
        return ecoLevel;
    }

    int thermalStatus() {
        return thermalStatus;
    }

    /** Ouvre la caméra voulue. */
    private void openCamera(boolean front) {
        video = new VideoSource(this, this);
        if (pendingPreview != null) video.setPreview(pendingPreview);
        try {
            video.start(front, prefs.width, prefs.height, prefs.fps, prefs.bitrate);
        } catch (Exception e) {
            status = "Caméra : " + e.getMessage();
            video.stop();
            video = null;
        }
    }

    /** Passe d'une caméra à l'autre sans toucher à la liaison avec le PC : l'appel continue. */
    void switchCamera(boolean front) {
        new Thread(() -> {
            synchronized (cameraLock) {
                if (!running) return;
                VideoSource old = video;
                video = null;
                if (old != null) old.stop();
                prefs.front = front;
                needKey = true;              // le PC repart de la première image de la nouvelle caméra
                openCamera(front);
                updateRotation();
            }
        }, "switch-camera").start();
    }

    private void startSources() {
        if (prefs.camera) {
            openCamera(prefs.front);
            // Le PC redresse l'image selon la façon dont on tient le téléphone.
            orientation = new OrientationEventListener(this) {
                @Override
                public void onOrientationChanged(int angle) {
                    if (angle == ORIENTATION_UNKNOWN) return;
                    int d = Math.abs(angle - deviceAngle) % 360;
                    if (Math.min(d, 360 - d) < 60) return;      // hystérésis : pas de bascule à 45°
                    deviceAngle = ((angle + 45) / 90 * 90) % 360;
                    updateRotation();
                }
            };
            if (orientation.canDetectOrientation()) orientation.enable();
            updateRotation();
        }
        if (prefs.mic) {
            audio = new AudioSource(this, this);
            audio.start(prefs.micMode, prefs.micGain);
        }
    }

    private void updateRotation() {
        VideoSource v = video;
        if (v == null) return;
        int dev = v.front ? -deviceAngle : deviceAngle;
        int r = (v.sensorOrientation + dev + 360) % 360;
        if (r != rotation) {
            rotation = r;
            sendRotation();
        }
    }

    private void sendRotation() {
        WsClient w = ws;
        if (w == null) return;
        try {
            w.sendText("rot:" + rotation);
        } catch (Exception ignored) {
        }
    }

    // --- connexion au PC (reconnexion automatique)
    private void netLoop() {
        while (running) {
            if (ws == null) {
                if (!prefs.linked()) {
                    status = "Pas encore relié au PC : « Connecter » sur le PC (câble) ou scanne le QR code (Wi-Fi).";
                } else {
                    try {
                        status = "Connexion à " + prefs.host + "…";
                        ws = WsClient.connect(prefs.host, prefs.tls, prefs.token, prefs.pin, this);
                        queue.clear();
                        queued.set(0);
                        needKey = true;
                        if (video != null) video.requestKey();
                        sendRotation();
                        if (webView != null) sendWeb(webView);
                        if (webMode != null) sendWeb(webMode);
                        sendWeb("{\"t\":\"eco\",\"level\":" + Math.max(0, ecoLevel) + "}");
                        status = "Connecté au PC (" + (prefs.tls ? "Wi-Fi" : "câble USB") + ").";
                    } catch (Exception e) {
                        status = "PC injoignable : " + e.getMessage() + ". Nouvel essai…";
                    }
                }
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                break;
            }
        }
    }

    private void sendLoop() {
        while (running) {
            try {
                byte[] m = queue.poll(200, TimeUnit.MILLISECONDS);
                if (m == null) continue;
                queued.addAndGet(-m.length);
                WsClient w = ws;
                if (w != null) w.send(m, 2);
            } catch (InterruptedException e) {
                break;
            } catch (Exception e) {
                dropConnection("envoi : " + e.getMessage());
            }
        }
    }

    private void dropConnection(String why) {
        WsClient w = ws;
        ws = null;
        if (w != null) w.close();
        if (running) status = "Liaison perdue (" + why + "), reconnexion…";
    }

    @Override
    public void onText(String text) {
        if ("key".equals(text) && video != null) video.requestKey();
        else if (text.startsWith("{")) {
            WebListener l = web;
            if (l != null) l.onWebText(text);
        }
    }

    @Override
    public void onBinary(byte[] data) {
        WebListener l = web;
        if (l != null && data.length > 1) l.onWebFrame(data[0], data);
    }

    /** Commande des onglets Navigation / Appel vers le PC (JSON). */
    void sendWeb(String json) {
        if (json.contains("\"t\":\"mode\"")) webMode = json;
        else if (json.contains("\"t\":\"view\"")) webView = json;
        WsClient w = ws;
        if (w == null) return;
        textOut.execute(() -> {
            try {
                w.sendText(json);
            } catch (Exception ignored) {
            }
        });
    }

    @Override
    public void onClosed(String reason) {
        dropConnection(reason);
    }

    @Override
    public void onVideo(byte[] data, boolean key) {
        if (ws == null) {
            needKey = true;
            return;
        }
        if (needKey && !key) return;   // sans image clé, le PC ne peut rien décoder
        // Réseau trop lent : on saute l'image (et on redemande une image clé) plutôt que
        // d'accumuler du retard.
        if (queued.get() > MAX_QUEUED) {
            droppedFrames++;
            needKey = true;
            video.requestKey();
            return;
        }
        needKey = false;
        queued.addAndGet(data.length);
        queue.offer(data);
        sentFrames++;
    }

    @Override
    public void onAudio(byte[] msg) {
        if (ws == null || queued.get() > MAX_QUEUED * 4) return;
        queued.addAndGet(msg.length);
        queue.offer(msg);
    }

    @Override
    public void onVideoError(String message) {
        status = "Caméra : " + message;
    }

    @Override
    public void onAudioError(String message) {
        status = "Micro : " + message;
    }

    // --- pour l'écran de l'appli
    void setPreview(Surface s) {
        pendingPreview = s;
        VideoSource v = video;
        if (v != null) v.setPreview(s);
    }

    void setTorch(boolean on) {
        if (video != null) video.setTorch(on);
    }

    VideoSource video() {
        return video;
    }

    float micLevel() {
        AudioSource a = audio;
        return a == null ? -120 : a.levelDb;
    }

    private Notification notification() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Envoi au PC", NotificationManager.IMPORTANCE_LOW));
        }
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        PendingIntent stop = PendingIntent.getService(this, 1, new Intent(this, StreamService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        String what = (prefs.camera ? "Caméra" : "") + (prefs.camera && prefs.mic ? " et micro" : prefs.mic ? "Micro" : "");
        Notification.Builder b = Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(this, CHANNEL) : new Notification.Builder(this);
        return b.setSmallIcon(R.drawable.ic_stat)
                .setContentTitle("PhoneCam")
                .setContentText(what + " envoyés au PC")
                .setOngoing(true)
                .setContentIntent(open)
                .addAction(new Notification.Action.Builder(null, "Arrêter", stop).build())
                .build();
    }

    @Override
    public void onDestroy() {
        running = false;
        instance = null;
        if (orientation != null) orientation.disable();
        if (netThread != null) netThread.interrupt();
        textOut.shutdownNow();
        if (sendThread != null) sendThread.interrupt();
        dropConnection("arrêt");
        if (video != null) video.stop();
        if (audio != null) audio.stop();
        video = null;
        audio = null;
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        if (wifiLock != null && wifiLock.isHeld()) wifiLock.release();
        if (thermal != null && Build.VERSION.SDK_INT >= 29) {
            ((PowerManager) getSystemService(POWER_SERVICE)).removeThermalStatusListener(thermal);
        }
        status = "Arrêté.";
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
