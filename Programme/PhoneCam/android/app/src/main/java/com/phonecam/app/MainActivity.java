package com.phonecam.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.util.Size;
import android.view.Gravity;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

/** Écran de l'appli : aperçu, réglages, démarrer / arrêter. L'envoi lui-même vit dans StreamService. */
public class MainActivity extends Activity implements SurfaceHolder.Callback {
    private static final int[][] SIZES = {{640, 480}, {1280, 720}, {1920, 1080}};
    private static final String[] SIZE_NAMES = {"480p", "720p", "1080p"};
    private static final int[] FPS = {24, 30, 60};
    private static final int[] BITRATES = {3_000_000, 6_000_000, 10_000_000, 16_000_000};
    private static final String[] BITRATE_NAMES = {"3 Mb/s", "6 Mb/s", "10 Mb/s", "16 Mb/s"};

    private Prefs prefs;
    private SurfaceView preview;
    private FrameLayout previewBox;
    private TextView status;
    private Button go;
    private boolean torch, autoStart;
    private SurfaceHolder holder;
    private Size shownSize;
    private StreamService attachedTo;     // service auquel l'aperçu est branché
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            refresh();
            ui.postDelayed(this, 500);
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        prefs = Prefs.load(this);
        limitRefreshRate();
        buildUi();
        handleLink(getIntent());
        askPermissions();
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        handleLink(intent);
    }

    /**
     * L'écran du S21 Ultra monte à 120 Hz dès que l'image bouge : l'appli n'affiche que 30 images/s
     * au plus, donc on demande 60 Hz (l'écran est l'une des plus grosses sources de chaleur).
     */
    private void limitRefreshRate() {
        android.view.Display d = getDisplay();
        if (d == null) return;
        android.view.Display.Mode cur = d.getMode(), best = null;
        for (android.view.Display.Mode m : d.getSupportedModes()) {
            if (m.getPhysicalWidth() != cur.getPhysicalWidth() || m.getPhysicalHeight() != cur.getPhysicalHeight()) continue;
            if (m.getRefreshRate() >= 59 && (best == null || m.getRefreshRate() < best.getRefreshRate())) best = m;
        }
        android.view.WindowManager.LayoutParams lp = getWindow().getAttributes();
        if (best != null) lp.preferredDisplayModeId = best.getModeId();
        lp.preferredRefreshRate = 60f;
        getWindow().setAttributes(lp);
    }

    /** Lien d'appairage envoyé par le PC : on le mémorise et on (re)démarre l'envoi. */
    private void handleLink(Intent intent) {
        Uri u = intent == null ? null : intent.getData();
        if (u != null && prefs.applyLink(u)) {
            prefs.save(this);
            autoStart = true;
            if (hasPermissions()) restart();
        }
    }

    private void buildUi() {
        int pad = dp(12);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF111111);

        previewBox = new FrameLayout(this);
        previewBox.setBackgroundColor(Color.BLACK);
        preview = new SurfaceView(this);
        previewBox.addView(preview, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT, Gravity.CENTER));
        preview.getHolder().addCallback(this);
        root.addView(previewBox, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(pad, pad, pad, pad);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(panel);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout row = row(panel);
        go = button(row, "Démarrer", v -> {
            if (StreamService.instance != null) StreamService.stop(this);
            else startStream();
        });
        go.setTypeface(Typeface.DEFAULT_BOLD);
        ((LinearLayout.LayoutParams) go.getLayoutParams()).weight = 2;
        button(row, "↺ Caméra", v -> flipCamera());
        button(row, "💡", v -> {
            torch = !torch;
            StreamService s = StreamService.instance;
            if (s != null) s.setTorch(torch);
        });

        row = row(panel);
        CheckBox cam = check(row, "📷 Caméra", prefs.camera);
        cam.setOnCheckedChangeListener((b, on) -> { prefs.camera = on; changed(); });
        CheckBox eco = check(row, "🔋 Éco", prefs.eco);
        eco.setOnCheckedChangeListener((b, on) -> { prefs.eco = on; changed(); });

        // Son du PC joué sur le téléphone, avec son volume (appliqué tout de suite).
        row = row(panel);
        CheckBox pc = check(row, "🔊 Son du PC", prefs.pcAudio);
        TextView pcVol = new TextView(this);
        pcVol.setTextColor(0xFFBBBBBB);
        pcVol.setText("Volume " + prefs.pcVolume + " %");
        row.addView(pcVol, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        SoftSlider vol = new SoftSlider(this, 0, 200, 5);
        vol.setValue(prefs.pcVolume);
        vol.onChange = v -> {
            prefs.pcVolume = v;
            pcVol.setText("Volume " + v + " %");
            pcAudioChanged();
        };
        panel.addView(vol, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        pc.setOnCheckedChangeListener((b, on) -> { prefs.pcAudio = on; pcAudioChanged(); });

        row = row(panel);
        spinner(row, "Résolution", SIZE_NAMES, indexOfSize(), i -> {
            prefs.width = SIZES[i][0];
            prefs.height = SIZES[i][1];
            changed();
        });
        spinner(row, "Images/s", new String[]{"24", "30", "60"}, indexOf(FPS, prefs.fps), i -> { prefs.fps = FPS[i]; changed(); });
        spinner(row, "Débit", BITRATE_NAMES, indexOf(BITRATES, prefs.bitrate), i -> { prefs.bitrate = BITRATES[i]; changed(); });

        status = new TextView(this);
        status.setTextColor(0xFFAAAAAA);
        status.setPadding(0, pad, 0, 0);
        panel.addView(status);

        TextView battery = new TextView(this);
        battery.setText("⚡ Si l'envoi s'arrête écran éteint : toucher ici pour ne pas limiter la batterie de PhoneCam.");
        battery.setTextColor(0xFF7FA7FF);
        battery.setPadding(0, pad, 0, 0);
        battery.setOnClickListener(v -> askBattery());
        panel.addView(battery);

        // Deux onglets : Caméra (ci-dessus) et Appel.
        bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setBackgroundColor(0xFF0B0B0F);
        web = new WebPages(this, () -> showTab(1), on -> bar.setVisibility(on ? View.GONE : View.VISIBLE),
                this::flipCamera);
        View[] pages = {root, web.callPage};
        LinearLayout outer = new LinearLayout(this);
        outer.setOrientation(LinearLayout.VERTICAL);
        outer.setBackgroundColor(0xFF0B0B0F);
        FrameLayout content = new FrameLayout(this);
        outer.addView(content, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        outer.addView(bar);
        String[] names = {"Caméra", "Appel"};
        for (int i = 0; i < 2; i++) {
            content.addView(pages[i], new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
            final int k = i;
            tabs[i] = button(bar, names[i], v -> showTab(k));
            tabs[i].setSingleLine(true);
            tabs[i].setTextSize(14);
            tabs[i].setBackground(null);
            tabs[i].setStateListAnimator(null);
        }
        this.pages = pages;
        showTab(0);
        // Android 15 dessine l'appli sous la barre d'état et les boutons de navigation : on réserve ces bords.
        outer.setOnApplyWindowInsetsListener((v, insets) -> {
            android.graphics.Insets b = insets.getInsets(android.view.WindowInsets.Type.systemBars()
                    | android.view.WindowInsets.Type.ime());
            v.setPadding(b.left, b.top, b.right, b.bottom);
            return android.view.WindowInsets.CONSUMED;
        });
        setContentView(outer);
    }

    private WebPages web;
    private LinearLayout bar;
    private StreamService modeSentTo;
    private View[] pages;
    private final Button[] tabs = new Button[2];
    private int tab;
    private static final String[] MODES = {"cam", "call"};

    private void showTab(int i) {
        tab = i;
        for (int k = 0; k < 2; k++) {
            pages[k].setVisibility(k == i ? View.VISIBLE : View.GONE);
            tabs[k].setTypeface(k == i ? Typeface.DEFAULT_BOLD : Typeface.DEFAULT);
            tabs[k].setTextColor(k == i ? 0xFFFFFC00 : 0xFF8E8E9A);
        }
        web.setMode(MODES[i]);
    }

    private void flipCamera() {
        prefs.front = !prefs.front;
        prefs.save(this);
        StreamService s = StreamService.instance;
        if (s != null) s.switchCamera(prefs.front);
    }

    private void changed() {
        prefs.save(this);
        if (StreamService.instance != null) restart();
    }

    /** Son du PC : sans redémarrer l'envoi (la caméra ne se coupe pas). */
    private void pcAudioChanged() {
        prefs.save(this);
        StreamService s = StreamService.instance;
        if (s != null) s.setPcAudio(prefs.pcAudio, prefs.pcVolume);
    }

    private void startStream() {
        if (!hasPermissions()) {
            autoStart = true;
            askPermissions();
            return;
        }
        if (!prefs.camera) {
            status.setText("Coche la caméra.");
            return;
        }
        StreamService.start(this);
    }

    private void restart() {
        StreamService.stop(this);
        ui.postDelayed(this::startStream, 400);
    }

    /**
     * La caméra n'accepte un aperçu qu'à une taille qu'elle sait produire : on fixe d'abord la
     * taille de la surface, puis on la branche une fois qu'elle a réellement cette taille.
     */
    private void attachPreview(StreamService s, Size sz) {
        if (holder == null || attachedTo == s) return;
        android.graphics.Rect r = holder.getSurfaceFrame();
        if (r.width() != sz.getWidth() || r.height() != sz.getHeight() || !holder.getSurface().isValid()) return;
        attachedTo = s;
        android.view.Surface surface = holder.getSurface();
        new Thread(() -> s.setPreview(surface)).start();
    }

    private void refresh() {
        StreamService s = StreamService.instance;
        go.setText(s != null ? "Arrêter" : "Démarrer");
        if (s != null && s != modeSentTo) {   // service (re)démarré : lui redire l'onglet affiché
            modeSentTo = s;
            web.setMode(MODES[tab]);
        }
        StringBuilder t = new StringBuilder();
        if (s == null) {
            t.append(prefs.linked() ? "Prêt. PC : " + prefs.host : "Pas encore relié au PC : « Connecter par câble » sur le PC, ou scanne son QR code.");
        } else {
            t.append(s.status);
            VideoSource v = s.video();
            if (v != null && v.size != null) {
                t.append("\n📷 ").append(v.size.getWidth()).append("×").append(v.size.getHeight())
                        .append(" · ").append(s.sentFrames * 2).append(" img/s");
                if (s.droppedFrames > 0) t.append(" · ").append(s.droppedFrames * 2).append(" sautées (réseau lent)");
                fitPreview(v.size);
                attachPreview(s, v.size);
            }
            s.sentFrames = 0;
            s.droppedFrames = 0;
            int eco = s.ecoLevel(), th = s.thermalStatus();
            String[] heat = {"froid", "normal", "tiède", "chaud", "très chaud", "brûlant", "brûlant", "brûlant"};
            t.append("\n🌡 Téléphone : ").append(heat[Math.max(0, Math.min(7, th))]);
            if (eco > 0) t.append(" · 🔋 économie ").append(eco == 1 ? "(24 img/s)" : "(20 img/s, chauffe forte)");
        }
        status.setText(t);
    }

    /** Aperçu au bon format : la caméra écrit à sa taille, l'écran tourne l'image tout seul. */
    private void fitPreview(Size sz) {
        if (sz.equals(shownSize) || holder == null) return;
        shownSize = sz;
        holder.setFixedSize(sz.getWidth(), sz.getHeight());
        int bw = previewBox.getWidth(), bh = previewBox.getHeight();
        if (bw == 0 || bh == 0) return;
        boolean portrait = getResources().getConfiguration().orientation == android.content.res.Configuration.ORIENTATION_PORTRAIT;
        float aspect = portrait ? (float) sz.getHeight() / sz.getWidth() : (float) sz.getWidth() / sz.getHeight();
        int w = bw, h = (int) (bw / aspect);
        if (h > bh) {
            h = bh;
            w = (int) (bh * aspect);
        }
        preview.setLayoutParams(new FrameLayout.LayoutParams(w, h, Gravity.CENTER));
    }

    @Override
    public void onConfigurationChanged(android.content.res.Configuration c) {
        super.onConfigurationChanged(c);
        shownSize = null;   // recalcule le cadre de l'aperçu
    }

    @Override
    protected void onResume() {
        super.onResume();
        StreamService.web = web;
        web.setMode(MODES[tab]);
        ui.post(tick);
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Appli en arrière-plan : le PC arrête d'envoyer l'écran et les vidéos (la caméra continue).
        web.setMode("cam");
        StreamService.web = null;
        ui.removeCallbacks(tick);
    }

    @Override
    public void surfaceCreated(SurfaceHolder h) {
        holder = h;
        shownSize = null;
        attachedTo = null;
    }

    @Override
    public void surfaceChanged(SurfaceHolder h, int format, int width, int height) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder h) {
        // L'appli passe en arrière-plan : la caméra continue, seulement vers l'encodeur.
        StreamService s = StreamService.instance;
        if (s != null) s.setPreview(null);
        holder = null;
        attachedTo = null;
    }

    // --- autorisations
    private String[] needed() {
        List<String> p = new ArrayList<>();
        p.add(Manifest.permission.CAMERA);
        if (Build.VERSION.SDK_INT >= 33) p.add(Manifest.permission.POST_NOTIFICATIONS);
        return p.toArray(new String[0]);
    }

    private boolean hasPermissions() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
;
    }

    private void askPermissions() {
        if (!hasPermissions()) requestPermissions(needed(), 1);
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        if (hasPermissions()) {
            if (autoStart) startStream();
        } else {
            status.setText("PhoneCam a besoin de la caméra et du micro pour fonctionner.");
        }
    }

    private void askBattery() {
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        if (pm.isIgnoringBatteryOptimizations(getPackageName())) {
            status.setText("La batterie n'est déjà pas limitée pour PhoneCam.");
            return;
        }
        try {
            startActivity(new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:" + getPackageName())));
        } catch (Exception e) {
            startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
        }
    }

    // --- petits utilitaires d'interface
    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    private LinearLayout row(LinearLayout parent) {
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.HORIZONTAL);
        r.setGravity(Gravity.CENTER_VERTICAL);
        parent.addView(r, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return r;
    }

    private Button button(LinearLayout parent, String text, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(l);
        parent.addView(b, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        return b;
    }

    private CheckBox check(LinearLayout parent, String text, boolean on) {
        CheckBox c = new CheckBox(this);
        c.setText(text);
        c.setTextColor(Color.WHITE);
        c.setChecked(on);
        parent.addView(c, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        return c;
    }

    interface OnPick {
        void pick(int index);
    }

    private void spinner(LinearLayout parent, String label, String[] items, int selected, OnPick l) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        TextView t = new TextView(this);
        t.setText(label);
        t.setTextColor(0xFFBBBBBB);
        t.setTextSize(12);
        box.addView(t);
        Spinner s = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, items) {
            @Override
            public View getView(int pos, View convert, ViewGroup parent) {   // texte lisible sur fond sombre
                TextView t = (TextView) super.getView(pos, convert, parent);
                t.setTextColor(Color.WHITE);
                return t;
            }
        };
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        s.setAdapter(adapter);
        s.setSelection(Math.max(0, selected), false);
        s.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            boolean first = true;

            @Override
            public void onItemSelected(AdapterView<?> a, View v, int pos, long id) {
                if (first) {     // appel initial lors de l'affichage : ce n'est pas un choix
                    first = false;
                    return;
                }
                l.pick(pos);
            }

            @Override
            public void onNothingSelected(AdapterView<?> a) {
            }
        });
        box.addView(s);
        boolean inRow = parent.getOrientation() == LinearLayout.HORIZONTAL;
        parent.addView(box, inRow ? new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                : new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
    }

    private int indexOfSize() {
        for (int i = 0; i < SIZES.length; i++) if (SIZES[i][0] == prefs.width) return i;
        return 1;
    }

    private static int indexOf(int[] a, int v) {
        for (int i = 0; i < a.length; i++) if (a[i] == v) return i;
        return 1;
    }
}
