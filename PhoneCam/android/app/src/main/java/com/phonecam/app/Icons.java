package com.phonecam.app;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.drawable.Drawable;

/** Icônes vectorielles (grille de 24 unités) : nettes à toutes les tailles, sans fichier à embarquer. */
final class Icons {
    static final String PHONE = "M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z";
    static final String VIDEO = "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z";
    static final String VIDEO_OFF = "M21 6.5l-4 4V7c0-.55-.45-1-1-1H9.82L21 17.18V6.5zM3.27 2L2 3.27 4.73 6H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.21 0 .39-.08.54-.18L19.73 21 21 19.73 3.27 2z";
    static final String MIC = "M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z";
    static final String MIC_OFF = "M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3L3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z";
    static final String VOLUME = "M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z";
    static final String HANGUP = "M12 9c-1.6 0-3.15.25-4.6.72v3.1c0 .39-.23.74-.56.9-.98.49-1.87 1.12-2.66 1.85-.18.18-.43.28-.7.28-.28 0-.53-.11-.71-.29L.29 13.08c-.18-.17-.29-.42-.29-.7 0-.28.11-.53.29-.71C3.34 8.78 7.46 7 12 7s8.66 1.78 11.71 4.67c.18.18.29.43.29.71 0 .28-.11.53-.29.71l-2.48 2.48c-.18.18-.43.29-.71.29-.27 0-.52-.11-.7-.28-.79-.74-1.69-1.36-2.67-1.85-.33-.16-.56-.5-.56-.9v-3.1C15.15 9.25 13.6 9 12 9z";
    static final String CLOSE = "M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z";
    static final String SPARKLE = "M19 9l1.25-2.75L23 5l-2.75-1.25L19 1l-1.25 2.75L15 5l2.75 1.25L19 9zm-7.5.5L9 4 6.5 9.5 1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5zM19 15l-1.25 2.75L15 19l2.75 1.25L19 23l1.25-2.75L23 19l-2.75-1.25L19 15z";
    static final String NONE = "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM4 12c0-4.42 3.58-8 8-8 1.85 0 3.55.63 4.9 1.69L5.69 16.9C4.63 15.55 4 13.85 4 12zm8 8c-1.85 0-3.55-.63-4.9-1.69L18.31 7.1C19.37 8.45 20 10.15 20 12c0 4.42-3.58 8-8 8z";
    static final String SETTINGS = "M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z";
    static final String ZOOM = "M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zM12 10h-2v2H9v-2H7V9h2V7h1v2h2v1z";
    static final String MOON = "M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-2.98 0-5.4-2.42-5.4-5.4 0-1.81.89-3.42 2.26-4.4-.44-.06-.9-.1-1.36-.1z";
    static final String BRUSH = "M7 14c-1.66 0-3 1.34-3 3 0 1.31-1.16 2-2 2 .92 1.22 2.49 2 4 2 2.21 0 4-1.79 4-4 0-1.66-1.34-3-3-3zm13.71-9.37l-1.34-1.34c-.39-.39-1.02-.39-1.41 0L9 12.25 11.75 15l8.96-8.96c.39-.39.39-1.03 0-1.41z";
    static final String SHIELD = "M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z";
    static final String FLIP = "M19 8l-4 4h3c0 3.31-2.69 6-6 6-1.01 0-1.97-.25-2.8-.7l-1.46 1.46C8.97 19.54 10.43 20 12 20c4.42 0 8-3.58 8-8h3l-4-4zM6 12c0-3.31 2.69-6 6-6 1.01 0 1.97.25 2.8.7l1.46-1.46C15.03 4.46 13.57 4 12 4c-4.42 0-8 3.58-8 8H1l4 4 4-4H6z";
    static final String CHECK = "M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z";

    private Icons() {
    }

    static Drawable of(String data, int color) {
        return new Glyph(parse(data), color);
    }

    /** Sous-ensemble des tracés SVG utilisé ici : M L H V C S Z, en absolu et en relatif. */
    private static Path parse(String d) {
        Path p = new Path();
        float[] pos = {0};              // curseur dans le texte, partagé avec number()
        int i = 0, n = d.length();
        float x = 0, y = 0, cx = 0, cy = 0, sx = 0, sy = 0;
        char op = 'M';
        while (true) {
            while (i < n && (d.charAt(i) == ' ' || d.charAt(i) == ',')) i++;
            if (i >= n) return p;
            if (Character.isLetter(d.charAt(i))) op = d.charAt(i++);
            boolean rel = Character.isLowerCase(op);
            float ox = rel ? x : 0, oy = rel ? y : 0;
            pos[0] = i;
            switch (Character.toUpperCase(op)) {
                case 'M':
                    x = ox + number(d, pos);
                    y = oy + number(d, pos);
                    p.moveTo(x, y);
                    sx = cx = x;
                    sy = cy = y;
                    op = rel ? 'l' : 'L';
                    break;
                case 'L':
                    x = ox + number(d, pos);
                    y = oy + number(d, pos);
                    p.lineTo(x, y);
                    cx = x;
                    cy = y;
                    break;
                case 'H':
                    cx = x = ox + number(d, pos);
                    cy = y;
                    p.lineTo(x, y);
                    break;
                case 'V':
                    cy = y = oy + number(d, pos);
                    cx = x;
                    p.lineTo(x, y);
                    break;
                case 'C':
                case 'S': {
                    boolean smooth = Character.toUpperCase(op) == 'S';
                    float x1 = smooth ? 2 * x - cx : ox + number(d, pos);
                    float y1 = smooth ? 2 * y - cy : oy + number(d, pos);
                    float x2 = ox + number(d, pos), y2 = oy + number(d, pos);
                    float nx = ox + number(d, pos), ny = oy + number(d, pos);
                    p.cubicTo(x1, y1, x2, y2, nx, ny);
                    cx = x2;
                    cy = y2;
                    x = nx;
                    y = ny;
                    break;
                }
                default:                // Z
                    p.close();
                    cx = x = sx;
                    cy = y = sy;
                    break;
            }
            i = (int) pos[0];
        }
    }

    private static float number(String d, float[] pos) {
        int i = (int) pos[0], n = d.length();
        while (i < n && (d.charAt(i) == ' ' || d.charAt(i) == ',')) i++;
        int start = i;
        boolean dot = false;
        if (i < n && d.charAt(i) == '-') i++;
        while (i < n) {
            char c = d.charAt(i);
            if (c == '.' && !dot) dot = true;
            else if (c < '0' || c > '9') break;
            i++;
        }
        pos[0] = i;
        return Float.parseFloat(d.substring(start, i));
    }

    private static final class Glyph extends Drawable {
        private final Path path;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        Glyph(Path path, int color) {
            this.path = path;
            paint.setColor(color);
        }

        @Override
        public void draw(Canvas c) {
            int save = c.save();
            c.translate(getBounds().left, getBounds().top);
            c.scale(getBounds().width() / 24f, getBounds().height() / 24f);
            c.drawPath(path, paint);
            c.restoreToCount(save);
        }

        @Override
        public void setAlpha(int alpha) {
            paint.setAlpha(alpha);
        }

        @Override
        public void setColorFilter(ColorFilter f) {
            paint.setColorFilter(f);
        }

        @Override
        public int getOpacity() {
            return PixelFormat.TRANSLUCENT;
        }
    }
}
