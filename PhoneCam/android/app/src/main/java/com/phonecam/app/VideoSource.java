package com.phonecam.app;

import android.annotation.SuppressLint;
import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.util.Size;
import android.view.Surface;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Caméra (Camera2) → encodeur H.264 matériel (MediaCodec), sans passer par le processeur :
 * la caméra écrit directement dans la surface d'entrée de l'encodeur. Un aperçu facultatif
 * (quand l'appli est à l'écran) est ajouté comme seconde sortie de la caméra.
 */
final class VideoSource {
    interface Sink {
        /** data : message prêt à envoyer (type 1 = image clé avec SPS/PPS, 2 = image delta). */
        void onVideo(byte[] data, boolean key);

        void onVideoError(String message);
    }

    private final Context ctx;
    private final Sink sink;
    private HandlerThread thread;
    private Handler handler;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private MediaCodec encoder;
    private Surface encoderSurface;
    private Surface preview;
    private byte[] config = new byte[0];      // SPS/PPS, recollés devant chaque image clé
    private Range<Integer> fpsRange;
    private Range<Integer>[] fpsRanges;
    private int fps, bitrate;
    private boolean torch;
    volatile Size size;
    volatile int sensorOrientation;
    volatile boolean front;

    VideoSource(Context ctx, Sink sink) {
        this.ctx = ctx.getApplicationContext();
        this.sink = sink;
    }

    @SuppressLint("MissingPermission")
    void start(boolean wantFront, int w, int h, int fps, int bitrate) throws Exception {
        thread = new HandlerThread("camera");
        thread.start();
        handler = new Handler(thread.getLooper());
        CameraManager cm = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
        String id = null;
        CameraCharacteristics cc = null;
        for (String c : cm.getCameraIdList()) {
            CameraCharacteristics ch = cm.getCameraCharacteristics(c);
            Integer facing = ch.get(CameraCharacteristics.LENS_FACING);
            boolean isFront = facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT;
            if (isFront == wantFront) {
                id = c;
                cc = ch;
                break;
            }
        }
        if (id == null) throw new Exception("caméra " + (wantFront ? "avant" : "arrière") + " introuvable");
        front = wantFront;
        Integer so = cc.get(CameraCharacteristics.SENSOR_ORIENTATION);
        sensorOrientation = so == null ? 0 : so;

        StreamConfigurationMap map = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        size = pickSize(map.getOutputSizes(MediaCodec.class), w, h);
        fpsRanges = cc.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        fpsRange = pickFps(fpsRanges, fps);
        this.fps = fps;
        this.bitrate = bitrate;

        encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
        try {   // Profil Baseline : pas d'images B, donc aucune image de retard à l'encodage.
            encoder.configure(format(bitrate, fps, true), null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        } catch (Exception e) {      // certains encodeurs refusent un profil imposé
            encoder.reset();
            encoder.configure(format(bitrate, fps, false), null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        }
        encoder.setCallback(new MediaCodec.Callback() {
            @Override
            public void onInputBufferAvailable(MediaCodec codec, int index) {
            }

            @Override
            public void onOutputBufferAvailable(MediaCodec codec, int index, MediaCodec.BufferInfo info) {
                try {
                    ByteBuffer b = codec.getOutputBuffer(index);
                    if (b != null && info.size > 0) {
                        b.position(info.offset);
                        b.limit(info.offset + info.size);
                        if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                            config = new byte[info.size];
                            b.get(config);
                        } else {
                            boolean key = (info.flags & MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0;
                            byte[] head = key ? config : new byte[0];
                            byte[] msg = new byte[1 + head.length + info.size];
                            msg[0] = (byte) (key ? 1 : 2);
                            System.arraycopy(head, 0, msg, 1, head.length);
                            b.get(msg, 1 + head.length, info.size);
                            sink.onVideo(msg, key);
                        }
                    }
                    codec.releaseOutputBuffer(index, false);
                } catch (IllegalStateException ignored) {   // encodeur arrêté entre-temps
                }
            }

            @Override
            public void onError(MediaCodec codec, MediaCodec.CodecException e) {
                sink.onVideoError("encodeur : " + e.getMessage());
            }

            @Override
            public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
            }
        }, handler);
        encoderSurface = encoder.createInputSurface();
        encoder.start();

        CountDownLatch opened = new CountDownLatch(1);
        final Exception[] err = {null};
        cm.openCamera(id, new CameraDevice.StateCallback() {
            @Override
            public void onOpened(CameraDevice c) {
                camera = c;
                opened.countDown();
                createSession();
            }

            @Override
            public void onDisconnected(CameraDevice c) {
                c.close();
                camera = null;
                sink.onVideoError("caméra prise par une autre application");
                opened.countDown();
            }

            @Override
            public void onError(CameraDevice c, int error) {
                c.close();
                camera = null;
                err[0] = new Exception("erreur caméra " + error);
                sink.onVideoError("erreur caméra " + error);
                opened.countDown();
            }
        }, handler);
        if (!opened.await(5, TimeUnit.SECONDS)) throw new Exception("la caméra ne s'ouvre pas");
        if (err[0] != null) throw err[0];
    }

    /** Ajoute ou retire l'aperçu. Bloque jusqu'à ce que la caméra n'écrive plus dans l'ancien. */
    void setPreview(Surface s) {
        if (handler == null) {
            preview = s;
            return;
        }
        CountDownLatch done = new CountDownLatch(1);
        handler.post(() -> {
            preview = s;
            createSession();
            done.countDown();
        });
        try {
            done.await(2, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
        }
    }

    /**
     * Économie d'énergie sans couper la caméra : 0 = normal, 1 = 24 img/s et débit −30 %,
     * 2 = 20 img/s et débit −50 %. La caméra et l'encodeur ralentissent, donc chauffent moins.
     */
    void setEco(int level) {
        if (handler == null) return;
        handler.post(() -> {
            int target = level == 0 ? fps : Math.min(fps, level == 1 ? 24 : 20);
            Range<Integer> r = pickFps(fpsRanges, target);
            if (r != null) fpsRange = r;
            repeat();
            MediaCodec e = encoder;
            if (e != null) {
                try {
                    Bundle b = new Bundle();
                    b.putInt(MediaCodec.PARAMETER_KEY_VIDEO_BITRATE,
                            (int) (bitrate * (level == 0 ? 1 : level == 1 ? 0.7 : 0.5)));
                    e.setParameters(b);
                } catch (IllegalStateException ignored) {
                }
            }
        });
    }

    void setTorch(boolean on) {
        if (handler == null) return;
        handler.post(() -> {
            torch = on;
            repeat();
        });
    }

    void requestKey() {
        MediaCodec e = encoder;
        if (e == null) return;
        try {
            Bundle b = new Bundle();
            b.putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0);
            e.setParameters(b);
        } catch (IllegalStateException ignored) {
        }
    }

    private void createSession() {
        if (camera == null) return;
        if (session != null) {
            session.close();
            session = null;
        }
        List<Surface> outs = new ArrayList<>();
        outs.add(encoderSurface);
        if (preview != null && preview.isValid()) outs.add(preview);
        try {
            camera.createCaptureSession(outs, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession s) {
                    session = s;
                    repeat();
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession s) {
                    sink.onVideoError("configuration de la caméra refusée");
                }
            }, handler);
        } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
            sink.onVideoError("caméra : " + e.getMessage());
        }
    }

    private void repeat() {
        if (camera == null || session == null) return;
        try {
            // Modèle « aperçu » plutôt qu'« enregistrement » : traitements rapides, aucune image
            // retenue. La stabilisation vidéo (qui garde plusieurs images d'avance) est coupée.
            CaptureRequest.Builder rb = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            rb.addTarget(encoderSurface);
            if (preview != null && preview.isValid()) rb.addTarget(preview);
            rb.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            rb.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
            rb.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
            rb.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_FAST);
            rb.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_FAST);
            if (fpsRange != null) rb.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange);
            rb.set(CaptureRequest.FLASH_MODE, torch ? CaptureRequest.FLASH_MODE_TORCH : CaptureRequest.FLASH_MODE_OFF);
            session.setRepeatingRequest(rb.build(), null, handler);
        } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
            sink.onVideoError("caméra : " + e.getMessage());
        }
    }

    void stop() {
        Handler h = handler;
        if (h != null) {
            CountDownLatch done = new CountDownLatch(1);
            h.post(() -> {
                try {
                    if (session != null) session.close();
                    if (camera != null) camera.close();
                } catch (Exception ignored) {
                }
                session = null;
                camera = null;
                done.countDown();
            });
            try {
                done.await(2, TimeUnit.SECONDS);
            } catch (InterruptedException ignored) {
            }
        }
        if (encoder != null) {
            try {
                encoder.stop();
            } catch (Exception ignored) {
            }
            encoder.release();
            encoder = null;
        }
        if (encoderSurface != null) encoderSurface.release();
        encoderSurface = null;
        if (thread != null) thread.quitSafely();
        thread = null;
        handler = null;
    }

    private MediaFormat format(int bitrate, int fps, boolean baseline) {
        MediaFormat f = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, size.getWidth(), size.getHeight());
        f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        f.setInteger(MediaFormat.KEY_BIT_RATE, bitrate);
        f.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
        f.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
        f.setInteger(MediaFormat.KEY_PRIORITY, 0);                 // temps réel
        f.setInteger(MediaFormat.KEY_OPERATING_RATE, fps);         // pas plus vite que nécessaire
        if (Build.VERSION.SDK_INT >= 30) f.setInteger(MediaFormat.KEY_LOW_LATENCY, 1);
        f.setInteger("latency", 1);                                 // l'encodeur rend chaque image aussitôt
        if (baseline) {
            f.setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileBaseline);
            f.setInteger(MediaFormat.KEY_LEVEL, size.getWidth() * size.getHeight() > 1280 * 720
                    ? MediaCodecInfo.CodecProfileLevel.AVCLevel41 : MediaCodecInfo.CodecProfileLevel.AVCLevel31);
        }
        return f;
    }

    private static Size pickSize(Size[] sizes, int w, int h) {
        Size best = sizes[0];
        long bestScore = Long.MAX_VALUE;
        for (Size s : sizes) {
            // Même format d'image d'abord (16:9), puis la plus petite taille au moins égale à la
            // demande (jamais en dessous : une image trop petite se verrait), sinon la plus proche.
            long aspect = Math.abs((long) s.getWidth() * h - (long) s.getHeight() * w) * 1000L / ((long) w * h);
            long area = (long) s.getWidth() * s.getHeight() - (long) w * h;
            long score = aspect * 1_000_000_000L + (area >= 0 ? area : 500_000_000L - area);
            if (score < bestScore) {
                bestScore = score;
                best = s;
            }
        }
        return best;
    }

    private static Range<Integer> pickFps(Range<Integer>[] ranges, int fps) {
        if (ranges == null) return null;
        Range<Integer> best = null;
        for (Range<Integer> r : ranges) {
            if (r.getUpper() != fps) continue;
            if (best == null || r.getLower() > best.getLower()) best = r;   // [30,30] plutôt que [15,30]
        }
        if (best != null) return best;
        for (Range<Integer> r : ranges) if (r.contains(fps)) return r;
        return null;
    }
}
