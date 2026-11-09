package com.phonecam.app;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;

/**
 * Micro → blocs de 20 ms (48 kHz, mono, int16), au format attendu par PhoneCam :
 * octet 4, fréquence (uint32 petit-boutiste), puis les échantillons.
 */
final class AudioSource {
    interface Sink {
        void onAudio(byte[] msg);

        void onAudioError(String message);
    }

    static final int RATE = 48000;
    private static final int BLOCK = 960;
    private final Context ctx;
    private final Sink sink;
    private volatile boolean running;
    private Thread thread;
    volatile float levelDb = -120;

    AudioSource(Context ctx, Sink sink) {
        this.ctx = ctx.getApplicationContext();
        this.sink = sink;
    }

    static final int MODE_CALL = 0, MODE_SPEAKER = 1, MODE_VIDEO = 2, MODE_RAW = 3;

    void start(int mode, int gainDb) {
        running = true;
        thread = new Thread(() -> loop(mode, gainDb), "micro");
        thread.setPriority(Thread.MAX_PRIORITY);
        thread.start();
    }

    @SuppressLint("MissingPermission")
    private void loop(int mode, int gainDb) {
        AudioManager am = (AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE);
        int source;
        boolean speaker = false;
        switch (mode) {
            case MODE_SPEAKER:
                // Appel en haut-parleur : le téléphone change de micros et de réglages (prise de
                // son « pièce », gain plus fort) pour entendre quelqu'un à plusieurs mètres.
                source = MediaRecorder.AudioSource.VOICE_COMMUNICATION;
                speaker = true;
                break;
            case MODE_VIDEO:
                // Réglage de l'appli Appareil photo : tous les micros, son naturel, sans réduction de bruit.
                source = MediaRecorder.AudioSource.CAMCORDER;
                break;
            case MODE_RAW:
                source = "true".equals(am.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED))
                        ? MediaRecorder.AudioSource.UNPROCESSED : MediaRecorder.AudioSource.MIC;
                break;
            default:   // appel à l'oreille : anti-écho, réduction de bruit, gain auto, micro du bas
                source = MediaRecorder.AudioSource.VOICE_COMMUNICATION;
        }
        if (speaker) routeToSpeaker(am, true);
        float gain = (float) Math.pow(10, gainDb / 20.0);
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord rec;
        try {
            rec = new AudioRecord(source, RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    Math.max(min, BLOCK * 2 * 4));
            if (rec.getState() != AudioRecord.STATE_INITIALIZED) throw new IllegalStateException("micro indisponible");
            rec.startRecording();
        } catch (Exception e) {
            sink.onAudioError("micro : " + e.getMessage());
            if (speaker) routeToSpeaker(am, false);
            return;
        }
        short[] pcm = new short[BLOCK];
        try {
            while (running) {
                int got = 0;
                while (got < BLOCK && running) {
                    int r = rec.read(pcm, got, BLOCK - got);
                    if (r < 0) throw new IllegalStateException("lecture du micro : " + r);
                    got += r;
                }
                if (!running) break;
                byte[] msg = new byte[5 + BLOCK * 2];
                msg[0] = 4;
                msg[1] = (byte) RATE;
                msg[2] = (byte) (RATE >> 8);
                msg[3] = (byte) (RATE >> 16);
                msg[4] = (byte) (RATE >> 24);
                int peak = 1;
                for (int i = 0; i < BLOCK; i++) {
                    short v = pcm[i];
                    if (gain != 1f) {   // gain ajouté, avec un écrêtage doux plutôt qu'une saturation sèche
                        float x = v * gain / 32768f;
                        x = Math.abs(x) < 0.7f ? x : Math.signum(x) * (0.7f + 0.3f * (float) Math.tanh((Math.abs(x) - 0.7f) / 0.3f));
                        v = (short) (x * 32767);
                    }
                    msg[5 + 2 * i] = (byte) v;
                    msg[6 + 2 * i] = (byte) (v >> 8);
                    peak = Math.max(peak, Math.abs((int) v));
                }
                levelDb = (float) (20 * Math.log10(peak / 32768.0));
                sink.onAudio(msg);
            }
        } catch (Exception e) {
            if (running) sink.onAudioError(e.getMessage());
        } finally {
            try {
                rec.stop();
            } catch (Exception ignored) {
            }
            rec.release();
            if (speaker) routeToSpeaker(am, false);
        }
    }

    /** Mode « appel en haut-parleur » (et retour à la normale à l'arrêt). */
    private static void routeToSpeaker(AudioManager am, boolean on) {
        try {
            if (on) {
                am.setMode(AudioManager.MODE_IN_COMMUNICATION);
                if (Build.VERSION.SDK_INT >= 31) {
                    for (AudioDeviceInfo d : am.getAvailableCommunicationDevices()) {
                        if (d.getType() == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER) {
                            am.setCommunicationDevice(d);
                            break;
                        }
                    }
                } else {
                    am.setSpeakerphoneOn(true);
                }
            } else {
                if (Build.VERSION.SDK_INT >= 31) am.clearCommunicationDevice();
                else am.setSpeakerphoneOn(false);
                am.setMode(AudioManager.MODE_NORMAL);
            }
        } catch (Exception ignored) {   // le son passe quand même, sans le réglage haut-parleur
        }
    }

    void stop() {
        running = false;
        if (thread != null) {
            try {
                thread.join(500);
            } catch (InterruptedException ignored) {
            }
        }
        thread = null;
        levelDb = -120;
    }
}
