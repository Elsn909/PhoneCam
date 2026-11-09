package com.phonecam.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.HapticFeedbackConstants;
import android.view.MotionEvent;
import android.view.View;

import java.util.function.IntConsumer;

/**
 * Curseur doux : il suit le déplacement du doigt à partir de sa position actuelle (toucher la piste ne
 * le fait pas sauter) et avance par crans, avec un petit retour de vibration à chacun.
 */
final class SoftSlider extends View {
    private static final int TRACK = 0x33FFFFFF, FILL = 0xFFFFFC00;
    private final int lo, hi, step;
    private final float density;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF rect = new RectF();
    private int value;
    private float exact, lastX;
    private boolean dragging;
    /** Appelé à chaque cran franchi par le doigt. */
    IntConsumer onChange = v -> {
    };

    SoftSlider(Context c, int lo, int hi, int step) {
        super(c);
        this.lo = lo;
        this.hi = hi;
        this.step = step;
        density = c.getResources().getDisplayMetrics().density;
        value = lo;
    }

    boolean isDragging() {
        return dragging;
    }

    void setValue(int v) {
        if (dragging) return;          // la main reste au doigt
        value = Math.max(lo, Math.min(hi, v));
        invalidate();
    }

    @Override
    protected void onMeasure(int w, int h) {
        setMeasuredDimension(MeasureSpec.getSize(w), (int) (44 * density));   // grande zone de toucher
    }

    private float thumbRadius() {
        return 12 * density;
    }

    @Override
    protected void onDraw(Canvas c) {
        float r = thumbRadius(), y = getHeight() / 2f, h = 3 * density, w = getWidth() - 2 * r;
        float x = r + w * (value - lo) / (hi - lo);
        paint.setColor(TRACK);
        rect.set(r, y - h, r + w, y + h);
        c.drawRoundRect(rect, h, h, paint);
        paint.setColor(FILL);
        rect.set(r, y - h, x, y + h);
        c.drawRoundRect(rect, h, h, paint);
        paint.setColor(0xFFFFFFFF);
        c.drawCircle(x, y, dragging ? r * 1.1f : r, paint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                dragging = true;
                exact = value;
                lastX = e.getX();
                getParent().requestDisallowInterceptTouchEvent(true);
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                float width = getWidth() - 2 * thumbRadius();
                exact += (e.getX() - lastX) / width * (hi - lo);
                exact = Math.max(lo, Math.min(hi, exact));
                lastX = e.getX();
                int stepped = lo + Math.round((exact - lo) / step) * step;
                if (stepped != value) {
                    value = stepped;
                    performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK);
                    onChange.accept(value);
                    invalidate();
                }
                return true;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                dragging = false;
                invalidate();
                return true;
            default:
                return false;
        }
    }
}
