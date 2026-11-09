package com.phonecam.app;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

/**
 * Son du PC joué sur le téléphone (message binaire de type 30 : PCM 16 bits stéréo 48 kHz, 20 ms).
 * Petite réserve de 60 ms contre les à-coups du réseau ; au-delà de 200 ms d'avance (horloges du PC
 * et du téléphone légèrement différentes), le retard est rattrapé en sautant des paquets.
 * Volume de 0 à 200 % : au-dessus de 100 %, gain numérique avec écrêtage doux.
 */
final class PcAudio {
    static final int KIND = 30;
    private static final int RATE = 48000, PRIME = 3, MAX = 10, KEEP = 4;
    private final LinkedBlockingDeque<byte[]> queue = new LinkedBlockingDeque<>();
    private volatile boolean running;
    private volatile int volume = 100;
    private Thread thread;
    volatile int played, skipped;

    void setVolume(int v) {
        volume = Math.max(0, Math.min(200, v));
    }

    synchronized void start() {
        if (running) return;
        running = true;
        queue.clear();
        thread = new Thread(this::loop, "son-du-pc");
        thread.setPriority(Thread.MAX_PRIORITY);
        thread.start();
    }

    synchronized void stop() {
        running = false;
        if (thread != null) thread.interrupt();
        thread = null;
        queue.clear();
    }

    boolean isRunning() {
        return running;
    }

    void feed(byte[] msg) {
        if (!running || msg.length < 5) return;
        queue.offer(msg);
        if (queue.size() > MAX) {                 // trop d'avance : on rattrape
            while (queue.size() > KEEP) {
                queue.pollFirst();
                skipped++;
            }
        }
    }

    private void loop() {
        int min = AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT);
        AudioTrack track;
        try {
            track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(RATE)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO).build())
                    .setBufferSizeInBytes(Math.max(min, 960 * 4 * 2))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                    .build();
        } catch (Exception e) {
            running = false;
            return;
        }
        track.play();
        short[] pcm = new short[0];
        boolean primed = false;
        try {
            while (running) {
                if (!primed) {
                    if (queue.size() < PRIME) {
                        Thread.sleep(5);
                        continue;
                    }
                    primed = true;
                }
                byte[] m = queue.poll(100, TimeUnit.MILLISECONDS);
                if (m == null) {                  // plus rien (le PC ne joue rien) : on se réamorce
                    primed = false;
                    continue;
                }
                int n = (m.length - 1) / 2;
                if (pcm.length != n) pcm = new short[n];
                ByteBuffer.wrap(m, 1, n * 2).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pcm);
                int vol = volume;
                if (vol != 100) {
                    float g = vol / 100f;
                    for (int i = 0; i < n; i++) {
                        float x = pcm[i] * g / 32768f;
                        if (g > 1f) x = (float) Math.tanh(x);       // pas de saturation dure
                        pcm[i] = (short) Math.max(-32768, Math.min(32767, Math.round(x * 32767f)));
                    }
                }
                track.write(pcm, 0, n);
                played++;
            }
        } catch (InterruptedException ignored) {
        } finally {
            try {
                track.stop();
            } catch (Exception ignored) {
            }
            track.release();
        }
    }
}
