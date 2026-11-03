package com.phonecam.app;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.media.MediaCodec;
import android.media.MediaFormat;
import android.view.Surface;
import android.view.TextureView;

import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * Une vidéo de l'appel : H.264 reçu du PC, décodé par le téléphone et affiché directement, image par
 * image. Chaque message : type, 1 si image clé, largeur et hauteur sur 2 octets, image.
 */
final class VideoTile implements TextureView.SurfaceTextureListener {
    private static final int HEADER = 6;
    final TextureView view;
    private final ArrayBlockingQueue<byte[]> queue = new ArrayBlockingQueue<>(20);
    private final Runnable askKey;
    private final MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
    private volatile Surface surface;
    private volatile boolean needKey = true;
    private volatile int width, height;
    private MediaCodec codec;
    private Surface codecSurface;

    /** askKey : demande au PC de repartir d'une image clé. */
    VideoTile(Context c, Runnable askKey) {
        this.askKey = askKey;
        view = new TextureView(c);
        view.setOpaque(false);
        view.setSurfaceTextureListener(this);
        view.addOnLayoutChangeListener((v, l, t, r, b, ol, ot, or, ob) -> fit());
        Thread t = new Thread(this::run, "video-tile");
        t.setDaemon(true);
        t.start();
    }

    /** Depuis le fil réseau : ne bloque jamais la réception. */
    void feed(byte[] data) {
        if (data.length <= HEADER) return;
        if (!queue.offer(data)) {
            queue.clear();
            needKey = true;
            askKey.run();
        }
    }

    private void run() {
        while (true) {
            try {
                decode(queue.take());
            } catch (InterruptedException e) {
                return;
            } catch (RuntimeException e) {   // décodeur perdu : on repart proprement de l'image clé suivante
                release();
                needKey = true;
                askKey.run();
            }
        }
    }

    private void decode(byte[] d) {
        boolean key = d[1] == 1;
        if (surface == null || surface != codecSurface) release();
        if (surface == null) {
            needKey = true;
            return;
        }
        if (needKey && !key) return;
        if (key) {
            int w = (d[2] & 255) << 8 | d[3] & 255, h = (d[4] & 255) << 8 | d[5] & 255;
            if (codec == null || w != width || h != height) {
                release();
                open(w, h);
            }
        }
        if (codec == null) return;
        needKey = false;
        int in = codec.dequeueInputBuffer(20_000);
        if (in >= 0) {
            ByteBuffer b = codec.getInputBuffer(in);
            b.clear();
            b.put(d, HEADER, d.length - HEADER);
            codec.queueInputBuffer(in, 0, d.length - HEADER, System.nanoTime() / 1000, 0);
        } else {
            needKey = true;
            askKey.run();
        }
        int out;
        while ((out = codec.dequeueOutputBuffer(info, 1000)) != MediaCodec.INFO_TRY_AGAIN_LATER) {
            if (out >= 0) codec.releaseOutputBuffer(out, true);
        }
    }

    private void open(int w, int h) {
        try {
            MediaFormat f = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h);
            f.setInteger(MediaFormat.KEY_LOW_LATENCY, 1);
            f.setInteger(MediaFormat.KEY_PRIORITY, 0);    // temps réel
            // Le flux ne porte pas ces informations : sans elles, le téléphone les devine selon la taille.
            f.setInteger(MediaFormat.KEY_COLOR_STANDARD, MediaFormat.COLOR_STANDARD_BT709);
            f.setInteger(MediaFormat.KEY_COLOR_RANGE, MediaFormat.COLOR_RANGE_LIMITED);
            f.setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO);
            codec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
            codec.configure(f, surface, null, 0);
            codec.start();
            codecSurface = surface;
            width = w;
            height = h;
            view.post(this::fit);
        } catch (java.io.IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private void release() {
        if (codec != null) {
            try {
                codec.stop();
            } catch (RuntimeException ignored) {
            }
            codec.release();
        }
        codec = null;
        codecSurface = null;
    }

    /** Remplit le cadre sans déformer, comme CENTER_CROP. */
    private void fit() {
        int vw = view.getWidth(), vh = view.getHeight(), w = width, h = height;
        if (vw == 0 || vh == 0 || w == 0 || h == 0) return;
        float s = Math.max(vw / (float) w, vh / (float) h);
        Matrix m = new Matrix();
        m.setScale(w * s / vw, h * s / vh, vw / 2f, vh / 2f);
        view.setTransform(m);
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture t, int w, int h) {
        surface = new Surface(t);
        needKey = true;
        askKey.run();
        fit();
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture t, int w, int h) {
        fit();
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture t) {
        surface = null;
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture t) {
    }
}
