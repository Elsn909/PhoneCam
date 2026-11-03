package com.phonecam.app;

import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Outline;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.text.TextUtils;
import android.util.Base64;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewOutlineProvider;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Onglet « Appel » : contacts à appeler, sonnerie, appel en cours en plein écran (deux vidéos,
 * filtres Snapchat, caméra, micro du PC). Le PC lit Snapchat Web et y rejoue les commandes.
 */
final class WebPages implements StreamService.WebListener {
    private static final int YELLOW = 0xFFFFFC00, BG = 0xFF0B0B0F, SURFACE = 0xFF17171D, RAISED = 0xFF25252D,
            LINE = 0x14FFFFFF, MUTED = 0xFF8E8E9A, GREEN = 0xFF30D158, RED = 0xFFFF453A, GLASS = 0x40FFFFFF,
            INK = 0xFF0B0B0F, SCRIM = 0xB3000000;
    private static final Typeface MEDIUM = Typeface.create("sans-serif-medium", Typeface.NORMAL);
    private static final Typeface HEAVY = Typeface.create("sans-serif", Typeface.BOLD);
    final View callPage;
    private final Activity a;
    private final Runnable showCall;
    private final Consumer<Boolean> fullScreen;
    private String mode = "cam";
    // écrans
    private final FrameLayout root, live, ringPicBox, sheet, settings;
    private final SoftSlider zoomSlider, darkSlider, blushSlider;
    private final android.widget.Switch guardSwitch;
    private final Map<SoftSlider, java.util.function.IntConsumer> settingShow = new HashMap<>();
    private final LinearLayout idle, contacts, ring, lensRow, audioBox, sheetList, micChip, camChip;
    private final TextView empty, ringName, ringText, peerName, peerTime, gainText, hint, audioMic;
    private final ImageView ringPic, peerPic;
    private final View answerBox, declineBox, ringEndBox, lensBar, halo;
    private final ImageView micKey, camKey;
    private final SeekBar volume;
    private final VideoTile remote, local;
    private final ObjectAnimator pulse;
    // état
    private final Map<String, Bitmap> avatars = new HashMap<>();
    private final Map<String, ImageView> rowPics = new HashMap<>();
    private final Map<String, View> lensViews = new HashMap<>();
    private JSONArray mics;
    private String micCurrent = "", peer = "", lens = "";
    private boolean swapped, calling, dragging, hasRemote, hasLocal;

    /** showCall : affiche l'onglet Appel quand un appel commence. fullScreen : masque les onglets en
     * appel. flip : passe de la caméra avant à l'arrière du téléphone, et inversement. */
    WebPages(Activity a, Runnable showCall, Consumer<Boolean> fullScreen, Runnable flip) {
        this.a = a;
        this.showCall = showCall;
        this.fullScreen = fullScreen;
        root = new FrameLayout(a);
        root.setBackgroundColor(BG);
        callPage = root;

        // --- pas d'appel : les contacts
        idle = new LinearLayout(a);
        idle.setOrientation(LinearLayout.VERTICAL);
        LinearLayout head = new LinearLayout(a);
        head.setGravity(Gravity.CENTER_VERTICAL);
        head.setPadding(dp(20), dp(18), dp(16), dp(14));
        TextView title = text("Appels", 30, Color.WHITE, HEAVY);
        head.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        key(head, Icons.SETTINGS, SURFACE, 44, v -> openSettings());
        idle.addView(head);
        hint = text("", 14, MUTED, null);
        hint.setPadding(dp(20), 0, dp(20), dp(10));
        idle.addView(hint);
        empty = text("Connexion à Snapchat…", 15, MUTED, null);
        empty.setGravity(Gravity.CENTER);
        empty.setPadding(dp(24), dp(48), dp(24), dp(10));
        idle.addView(empty);
        ScrollView sc = new ScrollView(a);
        sc.setOverScrollMode(View.OVER_SCROLL_NEVER);
        sc.setVerticalScrollBarEnabled(false);
        contacts = new LinearLayout(a);
        contacts.setOrientation(LinearLayout.VERTICAL);
        contacts.setPadding(dp(14), 0, dp(14), dp(14));
        sc.addView(contacts);
        idle.addView(sc, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        root.addView(idle, full());

        // --- ça sonne
        ring = new LinearLayout(a);
        ring.setOrientation(LinearLayout.VERTICAL);
        ring.setGravity(Gravity.CENTER_HORIZONTAL);
        ring.setBackground(new GradientDrawable(GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{0xFF2B2640, 0xFF15141D, BG}));
        ring.setVisibility(View.GONE);
        ring.addView(new View(a), new LinearLayout.LayoutParams(1, 0, 1f));
        ringPicBox = new FrameLayout(a);
        halo = new View(a);
        GradientDrawable haloShape = new GradientDrawable();
        haloShape.setShape(GradientDrawable.OVAL);
        haloShape.setColor(0x33FFFC00);
        halo.setBackground(haloShape);
        ringPicBox.addView(halo, new FrameLayout.LayoutParams(dp(150), dp(150), Gravity.CENTER));
        ringPic = avatar(150);
        ringPicBox.addView(ringPic, new FrameLayout.LayoutParams(dp(150), dp(150), Gravity.CENTER));
        ring.addView(ringPicBox, new LinearLayout.LayoutParams(dp(230), dp(230)));
        ringName = text("", 30, Color.WHITE, HEAVY);
        ringName.setPadding(dp(24), dp(8), dp(24), dp(6));
        ringName.setGravity(Gravity.CENTER);
        ring.addView(ringName);
        ringText = text("", 16, 0xCCFFFFFF, null);
        ringText.setGravity(Gravity.CENTER);
        ring.addView(ringText);
        ring.addView(new View(a), new LinearLayout.LayoutParams(1, 0, 1.2f));
        LinearLayout ringKeys = new LinearLayout(a);
        ringKeys.setGravity(Gravity.CENTER);
        ringKeys.setPadding(0, 0, 0, dp(56));
        declineBox = labelled(ringKeys, Icons.HANGUP, RED, "Refuser", v -> send(obj("t", "call", "a", "decline")));
        answerBox = labelled(ringKeys, Icons.PHONE, GREEN, "Décrocher", v -> send(obj("t", "call", "a", "answer")));
        ringEndBox = labelled(ringKeys, Icons.HANGUP, RED, "Annuler", v -> send(obj("t", "call", "a", "hangup")));
        ring.addView(ringKeys);
        root.addView(ring, full());
        pulse = ObjectAnimator.ofPropertyValuesHolder(halo,
                PropertyValuesHolder.ofFloat(View.SCALE_X, 1f, 1.5f),
                PropertyValuesHolder.ofFloat(View.SCALE_Y, 1f, 1.5f),
                PropertyValuesHolder.ofFloat(View.ALPHA, 1f, 0f));
        pulse.setDuration(1500);
        pulse.setRepeatCount(ValueAnimator.INFINITE);

        // --- en appel : plein écran
        live = new FrameLayout(a);
        live.setBackgroundColor(0xFF000000);
        live.setVisibility(View.INVISIBLE);    // pas GONE : les vidéos gardent leur surface, prêtes au décroché
        peerPic = avatar(150);
        live.addView(peerPic, new FrameLayout.LayoutParams(dp(150), dp(150), Gravity.CENTER));
        remote = new VideoTile(a, () -> send(obj("t", "key", "k", "R")));
        local = new VideoTile(a, () -> send(obj("t", "key", "k", "L")));
        for (VideoTile t : new VideoTile[]{remote, local}) {
            live.addView(t.view, full());
            t.view.setOnClickListener(v -> {
                if (v.getTranslationZ() > 0) {   // seule la petite vidéo inverse
                    swapped = !swapped;
                    layoutVideos();
                }
            });
        }

        View topShade = shade(GradientDrawable.Orientation.TOP_BOTTOM);
        live.addView(topShade, over(new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(150), Gravity.TOP), topShade));
        View bottomShade = shade(GradientDrawable.Orientation.BOTTOM_TOP);
        live.addView(bottomShade, over(new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(260), Gravity.BOTTOM), bottomShade));
        topShade.setElevation(dp(1));              // sous la petite vidéo
        bottomShade.setElevation(dp(1));

        LinearLayout top = new LinearLayout(a);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setPadding(dp(18), dp(14), dp(136), 0);
        peerName = text("", 20, Color.WHITE, HEAVY);
        peerName.setSingleLine(true);
        peerName.setEllipsize(TextUtils.TruncateAt.END);
        peerName.setShadowLayer(6, 0, 1, 0x80000000);
        top.addView(peerName);
        peerTime = text("", 14, 0xE6FFFFFF, null);
        peerTime.setShadowLayer(6, 0, 1, 0x80000000);
        top.addView(peerTime);
        micChip = chip(Icons.MIC_OFF, "Micro coupé");
        camChip = chip(Icons.VIDEO_OFF, "Caméra coupée");
        top.addView(micChip);
        top.addView(camChip);
        live.addView(top, over(new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP), top));

        // Retourner la caméra : sous la petite vidéo, comme dans Snapchat.
        LinearLayout flipBox = new LinearLayout(a);
        ImageView flipKey = key(flipBox, Icons.FLIP, GLASS, 46, v -> {
            flip.run();
            v.animate().rotationBy(180).setDuration(250).start();
        });
        flipKey.setContentDescription("Retourner la caméra");
        FrameLayout.LayoutParams flp = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP | Gravity.END);
        flp.setMargins(0, dp(184), dp(39), 0);
        live.addView(flipBox, over(flp, flipBox));

        LinearLayout bottom = new LinearLayout(a);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setPadding(0, 0, 0, dp(22));

        audioBox = new LinearLayout(a);
        audioBox.setOrientation(LinearLayout.VERTICAL);
        audioBox.setBackground(fill(0xE61C1C22, 22));
        audioBox.setPadding(dp(16), dp(12), dp(12), dp(10));
        audioBox.setVisibility(View.GONE);
        LinearLayout volumeRow = new LinearLayout(a);
        volumeRow.setGravity(Gravity.CENTER_VERTICAL);
        volumeRow.addView(icon(Icons.VOLUME, Color.WHITE, 20));
        volume = new SeekBar(a);
        volume.setMax(36);                         // −24 … +12 dB
        volume.setProgressTintList(ColorStateList.valueOf(YELLOW));
        volume.setThumbTintList(ColorStateList.valueOf(YELLOW));
        volume.setProgressBackgroundTintList(ColorStateList.valueOf(0x55FFFFFF));
        gainText = text("0 dB", 13, Color.WHITE, MEDIUM);
        volume.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar s, int v, boolean user) {
                gainText.setText(gainLabel(v - 24));
                if (user) send(obj("t", "gain", "db", v - 24));
            }

            @Override
            public void onStartTrackingTouch(SeekBar s) {
                dragging = true;
            }

            @Override
            public void onStopTrackingTouch(SeekBar s) {
                dragging = false;
            }
        });
        volumeRow.addView(volume, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        gainText.setMinWidth(dp(50));
        gainText.setGravity(Gravity.END);
        volumeRow.addView(gainText);
        audioBox.addView(volumeRow);
        LinearLayout micRow = new LinearLayout(a);
        micRow.setGravity(Gravity.CENTER_VERTICAL);
        micRow.setPadding(0, dp(12), 0, dp(6));
        micRow.addView(icon(Icons.MIC, Color.WHITE, 20));
        audioMic = text("Micro de l'appel", 14, Color.WHITE, MEDIUM);
        audioMic.setPadding(dp(14), 0, 0, 0);
        audioMic.setSingleLine(true);
        audioMic.setEllipsize(TextUtils.TruncateAt.END);
        micRow.addView(audioMic, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        micRow.setBackground(press(null, 12));
        micRow.setOnClickListener(v -> openMics());
        audioBox.addView(micRow);
        LinearLayout.LayoutParams vlp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        vlp.setMargins(dp(14), 0, dp(14), dp(10));
        bottom.addView(audioBox, vlp);

        HorizontalScrollView lensScroll = new HorizontalScrollView(a);
        lensScroll.setHorizontalScrollBarEnabled(false);
        lensScroll.setOverScrollMode(View.OVER_SCROLL_NEVER);
        lensRow = new LinearLayout(a);
        lensRow.setPadding(dp(10), dp(6), dp(10), dp(12));
        lensRow.setGravity(Gravity.CENTER_VERTICAL);
        lensScroll.addView(lensRow);
        lensBar = lensScroll;
        lensBar.setVisibility(View.GONE);
        bottom.addView(lensBar);
        onLenses(null);

        LinearLayout keys = new LinearLayout(a);
        keys.setGravity(Gravity.CENTER);
        camKey = key(keys, Icons.VIDEO, GLASS, 48, v -> send(obj("t", "call", "a", "camera")));
        micKey = key(keys, Icons.MIC, GLASS, 48, v -> send(obj("t", "call", "a", "mic")));
        key(keys, Icons.VOLUME, GLASS, 48, v -> {
            lensBar.setVisibility(View.GONE);
            toggle(audioBox);
        });
        key(keys, Icons.SPARKLE, GLASS, 48, v -> {
            audioBox.setVisibility(View.GONE);
            toggle(lensBar);
        });
        key(keys, Icons.SETTINGS, GLASS, 48, v -> openSettings());
        key(keys, Icons.HANGUP, RED, 56, v -> send(obj("t", "call", "a", "hangup")));
        bottom.addView(keys);
        live.addView(bottom, over(new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM), bottom));
        root.addView(live, full());
        layoutVideos();

        // --- liste des micros du PC, par-dessus tout
        sheet = new FrameLayout(a);
        sheet.setBackgroundColor(0x99000000);
        sheet.setVisibility(View.GONE);
        sheet.setOnClickListener(v -> sheet.setVisibility(View.GONE));
        LinearLayout card = new LinearLayout(a);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setClickable(true);
        card.setBackground(fill(SURFACE, 26));
        card.setPadding(dp(8), dp(18), dp(8), dp(20));
        TextView sheetTitle = text("Micro de l'appel", 20, Color.WHITE, HEAVY);
        sheetTitle.setPadding(dp(14), 0, 0, dp(10));
        card.addView(sheetTitle);
        ScrollView ss = new ScrollView(a);
        sheetList = new LinearLayout(a);
        sheetList.setOrientation(LinearLayout.VERTICAL);
        ss.addView(sheetList);
        card.addView(ss, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        FrameLayout.LayoutParams clp = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM);
        clp.setMargins(dp(8), dp(80), dp(8), dp(8));
        sheet.addView(card, clp);
        sheet.setElevation(dp(20));
        root.addView(sheet, full());

        // --- paramètres : quelques réglages de PhoneCam sur le PC
        settings = new FrameLayout(a);
        settings.setBackgroundColor(0x99000000);
        settings.setVisibility(View.GONE);
        settings.setOnClickListener(v -> settings.setVisibility(View.GONE));
        LinearLayout box = new LinearLayout(a);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setClickable(true);
        box.setBackground(fill(SURFACE, 26));
        box.setPadding(dp(18), dp(20), dp(18), dp(20));
        LinearLayout boxHead = new LinearLayout(a);
        boxHead.setGravity(Gravity.CENTER_VERTICAL);
        boxHead.setPadding(0, 0, 0, dp(8));
        boxHead.addView(text("Paramètres", 20, Color.WHITE, HEAVY),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        key(boxHead, Icons.CLOSE, RAISED, 34, v -> settings.setVisibility(View.GONE));
        box.addView(boxHead);
        zoomSlider = setting(box, Icons.ZOOM, "Zoom", "zoom", 100, 250, 5,
                v -> v == 100 ? "Aucun" : String.format(java.util.Locale.FRANCE, "×%.2f", v / 100f));
        darkSlider = setting(box, Icons.MOON, "Pièce sombre", "dark", 0, 100, 5, v -> v == 0 ? "Désactivée" : v + " %");
        blushSlider = setting(box, Icons.BRUSH, "Blush", "blush", 0, 100, 5, v -> v == 0 ? "Aucun" : v + " %");

        LinearLayout guardRow = new LinearLayout(a);
        guardRow.setGravity(Gravity.CENTER_VERTICAL);
        guardRow.setPadding(0, dp(14), 0, 0);
        guardRow.addView(badge(Icons.SHIELD));
        LinearLayout guardText = new LinearLayout(a);
        guardText.setOrientation(LinearLayout.VERTICAL);
        guardText.setPadding(dp(12), 0, dp(8), 0);
        guardText.addView(text("Anti-fuite", 15, Color.WHITE, MEDIUM));
        guardText.addView(text("Fige l'image avant que ton vrai visage apparaisse", 12, MUTED, null));
        guardRow.addView(guardText, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        guardSwitch = new android.widget.Switch(a);
        guardSwitch.setThumbTintList(ColorStateList.valueOf(Color.WHITE));
        guardSwitch.setTrackTintList(new ColorStateList(
                new int[][]{{android.R.attr.state_checked}, {}}, new int[]{YELLOW, 0x55FFFFFF}));
        guardSwitch.setOnClickListener(v -> sendSetting("guard", guardSwitch.isChecked()));
        guardRow.addView(guardSwitch);
        box.addView(guardRow);

        FrameLayout.LayoutParams blp = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM);
        blp.setMargins(dp(8), dp(80), dp(8), dp(8));
        settings.addView(box, blp);
        settings.setElevation(dp(20));
        root.addView(settings, full());
    }

    /** Une ligne du menu : pastille, titre, valeur, et le curseur dessous. */
    private SoftSlider setting(LinearLayout parent, String path, String title, String key, int lo, int hi, int step,
                               java.util.function.IntFunction<String> fmt) {
        LinearLayout head = new LinearLayout(a);
        head.setGravity(Gravity.CENTER_VERTICAL);
        head.setPadding(0, dp(10), 0, 0);
        head.addView(badge(path));
        TextView name = text(title, 15, Color.WHITE, MEDIUM);
        name.setPadding(dp(12), 0, 0, 0);
        head.addView(name, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        TextView shown = text(fmt.apply(lo), 14, MUTED, MEDIUM);
        head.addView(shown);
        parent.addView(head);
        SoftSlider s = new SoftSlider(a, lo, hi, step);
        s.onChange = v -> {
            shown.setText(fmt.apply(v));
            sendSetting(key, v);
        };
        parent.addView(s, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        settingShow.put(s, v -> shown.setText(fmt.apply(Math.max(lo, Math.min(hi, v)))));
        return s;
    }

    private View badge(String path) {
        ImageView v = new ImageView(a);
        v.setImageDrawable(Icons.of(path, Color.WHITE));
        v.setPadding(dp(8), dp(8), dp(8), dp(8));
        v.setBackground(disc(RAISED));
        v.setLayoutParams(new LinearLayout.LayoutParams(dp(36), dp(36)));
        return v;
    }

    private void sendSetting(String key, Object value) {
        send(obj("t", "pc", "key", key, "value", value));
    }

    private void openSettings() {
        send(obj("t", "pc"));
        settings.setVisibility(View.VISIBLE);
    }

    private void showSetting(SoftSlider s, int v) {
        if (s.isDragging()) return;
        s.setValue(v);
        settingShow.get(s).accept(v);
    }

    /** Valeurs de PhoneCam : elles suivent aussi ce qui change sur le PC. */
    private void onSettings(JSONObject o) {
        showSetting(zoomSlider, o.optInt("zoom", 100));
        showSetting(darkSlider, o.optInt("dark"));
        showSetting(blushSlider, o.optInt("blush"));
        guardSwitch.setChecked(o.optBoolean("guard"));
    }

    private <T extends ViewGroup.LayoutParams> T over(T lp, View v) {
        v.setElevation(dp(10));                     // au-dessus des deux vidéos
        v.setOutlineProvider(null);                 // sans ombre
        return lp;
    }

    /** Une vidéo en plein écran, l'autre en petit en haut à droite ; le toucher les échange. */
    private void layoutVideos() {
        VideoTile bigT = swapped ? local : remote, smallT = swapped ? remote : local;
        bigT.view.setLayoutParams(full());
        bigT.view.setTranslationZ(0);
        bigT.view.setClipToOutline(false);
        FrameLayout.LayoutParams slp = new FrameLayout.LayoutParams(dp(112), dp(160), Gravity.TOP | Gravity.END);
        slp.setMargins(0, dp(14), dp(14), 0);
        smallT.view.setLayoutParams(slp);
        smallT.view.setTranslationZ(dp(6));
        smallT.view.setOutlineProvider(rounded(18));
        smallT.view.setClipToOutline(true);
        smallT.view.setContentDescription("Inverser les deux vidéos");
        remote.view.setVisibility(hasRemote ? View.VISIBLE : View.INVISIBLE);
        local.view.setVisibility(hasLocal ? View.VISIBLE : View.INVISIBLE);
    }

    private static void toggle(View v) {
        v.setVisibility(v.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
    }

    // ------------------------------------------------------------ envoi au PC
    private void send(JSONObject o) {
        StreamService s = StreamService.instance;
        if (s != null) s.sendWeb(o.toString());
    }

    static JSONObject obj(Object... kv) {
        JSONObject o = new JSONObject();
        try {
            for (int i = 0; i + 1 < kv.length; i += 2) o.put((String) kv[i], kv[i + 1]);
        } catch (JSONException ignored) {
        }
        return o;
    }

    /** Onglet affiché : le PC n'envoie les vidéos que pour l'onglet Appel. */
    void setMode(String m) {
        mode = m;
        send(obj("t", "mode", "m", m));
        boolean off = StreamService.instance == null;
        hint.setText(off ? "Démarre d'abord l'envoi dans l'onglet Caméra." : "");
        hint.setVisibility(off ? View.VISIBLE : View.GONE);
    }

    // ------------------------------------------------------------ ce qui arrive du PC
    @Override
    public void onWebFrame(int kind, byte[] data) {
        if (kind == 21) remote.feed(data);
        else if (kind == 22) local.feed(data);
    }

    @Override
    public void onWebText(String json) {
        JSONObject o;
        try {
            o = new JSONObject(json);
        } catch (JSONException e) {
            return;
        }
        a.runOnUiThread(() -> {
            String t = o.optString("t");
            if ("web".equals(t)) toast(o.optString("status"));
            else if ("contacts".equals(t)) onContacts(o.optJSONArray("rows"));
            else if ("avatar".equals(t)) onAvatar(o.optString("name"), o.optString("b64"));
            else if ("lenses".equals(t)) onLenses(o.optJSONArray("items"));
            else if ("mics".equals(t)) onMics(o);
            else if ("pcset".equals(t)) onSettings(o);
            else if ("call".equals(t)) onCall(o);
        });
    }

    private void toast(String s) {
        if (!s.isEmpty()) Toast.makeText(a, s, Toast.LENGTH_SHORT).show();
    }

    private void onCall(JSONObject o) {
        boolean incoming = o.optBoolean("incoming"), inCall = o.optBoolean("inCall"), isLive = o.optBoolean("live");
        peer = o.optString("peer");
        boolean ringing = incoming || (inCall && !isLive);
        idle.setVisibility(!incoming && !inCall ? View.VISIBLE : View.GONE);
        ring.setVisibility(ringing ? View.VISIBLE : View.GONE);
        live.setVisibility(isLive ? View.VISIBLE : View.INVISIBLE);
        answerBox.setVisibility(incoming ? View.VISIBLE : View.GONE);
        declineBox.setVisibility(incoming ? View.VISIBLE : View.GONE);
        ringEndBox.setVisibility(incoming ? View.GONE : View.VISIBLE);
        if (ringing && !pulse.isStarted()) pulse.start();
        else if (!ringing) pulse.cancel();
        ringName.setText(peer.isEmpty() ? "Appel" : peer);
        ringText.setText(incoming ? "t'appelle…" : "Appel en cours…");
        setPic(ringPic, peer);
        setPic(peerPic, peer);
        peerName.setText(peer);
        peerTime.setText(o.optString("time"));

        boolean muted = o.optBoolean("muted"), camOff = o.optBoolean("camOff");
        micChip.setVisibility(muted ? View.VISIBLE : View.GONE);
        camChip.setVisibility(camOff ? View.VISIBLE : View.GONE);
        setKey(micKey, muted ? Icons.MIC_OFF : Icons.MIC, muted);
        setKey(camKey, camOff ? Icons.VIDEO_OFF : Icons.VIDEO, camOff);
        if (!dragging) volume.setProgress(o.optInt("gain") + 24);
        hasRemote = o.optBoolean("remote");
        hasLocal = o.optBoolean("local");
        layoutVideos();

        boolean now = incoming || inCall;
        if (incoming && !calling) buzz();
        if (now && !calling && !"call".equals(mode)) showCall.run();   // une seule fois par appel
        if (!now && calling) {
            swapped = false;
            lens = "";
            audioBox.setVisibility(View.GONE);
            lensBar.setVisibility(View.GONE);
            sheet.setVisibility(View.GONE);
            settings.setVisibility(View.GONE);
        }
        calling = now;
        fullScreen.accept(now);
    }

    private void onContacts(JSONArray rows) {
        contacts.removeAllViews();
        rowPics.clear();
        int n = rows == null ? 0 : rows.length();
        empty.setVisibility(n == 0 ? View.VISIBLE : View.GONE);
        for (int k = 0; k < n; k++) {
            JSONObject o = rows.optJSONObject(k);
            if (o != null) addContact(o);
        }
    }

    private void onAvatar(String name, String b64) {
        byte[] d = Base64.decode(b64, Base64.DEFAULT);
        Bitmap b = BitmapFactory.decodeByteArray(d, 0, d.length);
        if (b == null) return;
        avatars.put(name, b);
        ImageView pic = rowPics.get(name);
        if (pic != null) setPic(pic, name);
        if (name.equals(peer)) {
            setPic(ringPic, peer);
            setPic(peerPic, peer);
        }
    }

    /** La photo si elle est arrivée, sinon l'initiale. */
    private void setPic(ImageView v, String name) {
        Bitmap b = avatars.get(name);
        if (b != null) v.setImageBitmap(b);
        else v.setImageDrawable(new Initial(name, a.getResources().getDisplayMetrics().density));
    }

    private void addContact(JSONObject o) {
        int i = o.optInt("i");
        String name = o.optString("name");
        LinearLayout row = new LinearLayout(a);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(12), dp(10), dp(10), dp(10));
        row.setBackground(fill(SURFACE, 20));
        ImageView pic = avatar(54);
        setPic(pic, name);
        rowPics.put(name, pic);
        row.addView(pic);
        LinearLayout col = new LinearLayout(a);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(dp(12), 0, dp(6), 0);
        TextView n = text(name, 16, Color.WHITE, MEDIUM);
        n.setSingleLine(true);
        n.setEllipsize(TextUtils.TruncateAt.END);
        col.addView(n);
        TextView sub = text(o.optString("sub"), 13, MUTED, null);
        sub.setSingleLine(true);
        sub.setEllipsize(TextUtils.TruncateAt.END);
        col.addView(sub);
        row.addView(col, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        key(row, Icons.PHONE, RAISED, 44, v -> send(obj("t", "dial", "i", i, "video", false)));
        ImageView video = key(row, Icons.VIDEO, YELLOW, 44, v -> send(obj("t", "dial", "i", i, "video", true)));
        video.setImageDrawable(Icons.of(Icons.VIDEO, INK));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.bottomMargin = dp(8);
        contacts.addView(row, lp);
    }

    private void onLenses(JSONArray items) {
        lensRow.removeAllViews();
        lensViews.clear();
        if (items == null || items.length() == 0) {
            TextView t = text("Les filtres arrivent dès que ta caméra est active.", 14, Color.WHITE, null);
            t.setPadding(dp(10), dp(8), dp(10), dp(8));
            lensRow.addView(t);
            return;
        }
        lensRow.addView(lensItem("", "Aucun", null));
        for (int k = 0; k < items.length(); k++) {
            JSONObject o = items.optJSONObject(k);
            if (o == null) continue;
            byte[] d = Base64.decode(o.optString("b64"), Base64.DEFAULT);
            lensRow.addView(lensItem(o.optString("name"), o.optString("name"), BitmapFactory.decodeByteArray(d, 0, d.length)));
        }
        markLens();
    }

    private View lensItem(String name, String caption, Bitmap thumb) {
        LinearLayout col = new LinearLayout(a);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setGravity(Gravity.CENTER_HORIZONTAL);
        col.setPadding(dp(5), 0, dp(5), 0);
        FrameLayout box = new FrameLayout(a);
        box.setPadding(dp(3), dp(3), dp(3), dp(3));
        ImageView pic = avatar(56);
        if (thumb != null) pic.setImageBitmap(thumb);
        else {
            pic.setImageDrawable(Icons.of(Icons.NONE, Color.WHITE));
            pic.setPadding(dp(13), dp(13), dp(13), dp(13));
        }
        box.addView(pic, new FrameLayout.LayoutParams(dp(56), dp(56)));
        col.addView(box);
        TextView t = text(caption, 11, Color.WHITE, null);
        t.setSingleLine(true);
        t.setEllipsize(TextUtils.TruncateAt.END);
        t.setMaxWidth(dp(66));
        t.setPadding(0, dp(3), 0, 0);
        t.setShadowLayer(4, 0, 1, Color.BLACK);
        col.addView(t);
        col.setContentDescription("Filtre " + caption);
        col.setOnClickListener(v -> {
            lens = name;
            markLens();
            send(obj("t", "lens", "name", name));
        });
        lensViews.put(name, box);
        return col;
    }

    private void markLens() {
        for (Map.Entry<String, View> e : lensViews.entrySet()) {
            boolean on = e.getKey().equals(lens);
            GradientDrawable d = new GradientDrawable();
            d.setShape(GradientDrawable.OVAL);
            d.setStroke(dp(2), on ? YELLOW : 0x40FFFFFF);
            e.getValue().setBackground(d);
        }
    }

    private void onMics(JSONObject o) {
        mics = o.optJSONArray("items");
        micCurrent = o.optString("current");
        audioMic.setText(micCurrent.isEmpty() ? "Micro de l'appel" : micCurrent);
        fillMics();
    }

    /** La liste vient du menu de Snapchat : le PC le lit à chaque ouverture. */
    private void openMics() {
        mics = null;
        fillMics();
        send(obj("t", "mics"));
        sheet.setVisibility(View.VISIBLE);
    }

    private void fillMics() {
        sheetList.removeAllViews();
        int n = mics == null ? 0 : mics.length();
        if (n == 0) {
            TextView t = text("Lecture de la liste de Snapchat…", 15, MUTED, null);
            t.setPadding(dp(14), dp(8), dp(14), dp(8));
            sheetList.addView(t);
        }
        for (int k = 0; k < n; k++) {
            String name = mics.optString(k);
            LinearLayout row = new LinearLayout(a);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setMinimumHeight(dp(54));
            row.setPadding(dp(14), dp(10), dp(10), dp(10));
            row.setBackground(press(null, 14));
            TextView t = text(name, 15, Color.WHITE, name.equals(micCurrent) ? MEDIUM : null);
            row.addView(t, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            if (name.equals(micCurrent)) row.addView(icon(Icons.CHECK, YELLOW, 22));
            row.setOnClickListener(v -> {
                micCurrent = name;
                fillMics();
                sheet.setVisibility(View.GONE);
                send(obj("t", "mics", "name", name));
            });
            sheetList.addView(row);
        }
    }

    private static String gainLabel(int db) {
        return db == 0 ? "0 dB" : (db > 0 ? "+" : "") + db + " dB";
    }

    private void buzz() {
        android.os.Vibrator v = a.getSystemService(android.os.Vibrator.class);
        if (v != null) v.vibrate(android.os.VibrationEffect.createWaveform(new long[]{0, 400, 250, 400, 250, 400}, -1));
    }

    // ------------------------------------------------------------ utilitaires
    private int dp(int v) {
        return (int) (v * a.getResources().getDisplayMetrics().density);
    }

    private FrameLayout.LayoutParams full() {
        return new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
    }

    private TextView text(String s, int size, int color, Typeface face) {
        TextView t = new TextView(a);
        t.setText(s);
        t.setTextSize(size);
        t.setTextColor(color);
        if (face != null) t.setTypeface(face);
        return t;
    }

    private GradientDrawable fill(int color, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(radius));
        return d;
    }

    /** Retour visuel au toucher. */
    private RippleDrawable press(GradientDrawable shape, int radius) {
        return new RippleDrawable(ColorStateList.valueOf(0x33FFFFFF), shape, fill(Color.WHITE, radius));
    }

    private ImageView icon(String path, int color, int size) {
        ImageView v = new ImageView(a);
        v.setImageDrawable(Icons.of(path, color));
        v.setLayoutParams(new LinearLayout.LayoutParams(dp(size), dp(size)));
        return v;
    }

    private View shade(GradientDrawable.Orientation o) {
        View v = new View(a);
        v.setBackground(new GradientDrawable(o, new int[]{SCRIM, Color.TRANSPARENT}));
        return v;
    }

    private LinearLayout chip(String path, String s) {
        LinearLayout c = new LinearLayout(a);
        c.setGravity(Gravity.CENTER_VERTICAL);
        c.setPadding(dp(10), dp(6), dp(12), dp(6));
        c.setBackground(fill(0xD9FF453A, 14));
        c.addView(icon(path, Color.WHITE, 16));
        TextView t = text(s, 13, Color.WHITE, MEDIUM);
        t.setPadding(dp(6), 0, 0, 0);
        c.addView(t);
        c.setVisibility(View.GONE);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(8);
        c.setLayoutParams(lp);
        return c;
    }

    private GradientDrawable disc(int color) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(color);
        return d;
    }

    /** Bouton rond à icône. */
    private ImageView key(LinearLayout parent, String path, int color, int size, View.OnClickListener l) {
        ImageView b = new ImageView(a);
        b.setImageDrawable(Icons.of(path, Color.WHITE));
        int p = dp(size) * 26 / 100;
        b.setPadding(p, p, p, p);
        b.setBackground(new RippleDrawable(ColorStateList.valueOf(0x55FFFFFF), disc(color), null));
        b.setClickable(true);
        b.setOnClickListener(l);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(size), dp(size));
        int m = dp(size > 48 ? 6 : 4);
        lp.setMargins(m, 0, m, 0);
        parent.addView(b, lp);
        return b;
    }

    /** État d'un bouton : blanc quand la fonction est coupée, bien visible. */
    private void setKey(ImageView b, String path, boolean off) {
        b.setImageDrawable(Icons.of(path, off ? INK : Color.WHITE));
        b.setBackground(new RippleDrawable(ColorStateList.valueOf(0x55FFFFFF), disc(off ? Color.WHITE : GLASS), null));
    }

    private View labelled(LinearLayout parent, String path, int color, String label, View.OnClickListener l) {
        LinearLayout col = new LinearLayout(a);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setGravity(Gravity.CENTER_HORIZONTAL);
        key(col, path, color, 72, l);
        TextView t = text(label, 13, 0xCCFFFFFF, MEDIUM);
        t.setPadding(0, dp(10), 0, 0);
        t.setGravity(Gravity.CENTER);
        col.addView(t);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(130), ViewGroup.LayoutParams.WRAP_CONTENT);
        parent.addView(col, lp);
        return col;
    }

    private ViewOutlineProvider rounded(int radius) {
        return new ViewOutlineProvider() {
            @Override
            public void getOutline(View view, Outline o) {
                o.setRoundRect(0, 0, view.getWidth(), view.getHeight(), dp(radius));
            }
        };
    }

    private ImageView avatar(int size) {
        ImageView v = new ImageView(a);
        v.setLayoutParams(new LinearLayout.LayoutParams(dp(size), dp(size)));
        v.setScaleType(ImageView.ScaleType.CENTER_CROP);
        v.setBackground(disc(RAISED));
        v.setOutlineProvider(new ViewOutlineProvider() {
            @Override
            public void getOutline(View view, Outline o) {
                o.setOval(0, 0, view.getWidth(), view.getHeight());
            }
        });
        v.setClipToOutline(true);
        return v;
    }

    /** Initiale du contact, le temps que sa photo arrive. */
    private static final class Initial extends android.graphics.drawable.Drawable {
        private final String letter;
        private final android.graphics.Paint paint = new android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG);

        Initial(String name, float density) {
            int cp = name.isEmpty() ? '?' : name.codePointAt(0);
            letter = new String(Character.toChars(Character.toUpperCase(cp)));
            paint.setColor(Color.WHITE);
            paint.setTypeface(MEDIUM);
            paint.setTextAlign(android.graphics.Paint.Align.CENTER);
        }

        @Override
        public void draw(android.graphics.Canvas c) {
            android.graphics.Rect r = getBounds();
            paint.setTextSize(r.height() * 0.42f);
            c.drawText(letter, r.exactCenterX(), r.exactCenterY() - (paint.ascent() + paint.descent()) / 2, paint);
        }

        @Override
        public void setAlpha(int alpha) {
            paint.setAlpha(alpha);
        }

        @Override
        public void setColorFilter(android.graphics.ColorFilter f) {
            paint.setColorFilter(f);
        }

        @Override
        public int getOpacity() {
            return android.graphics.PixelFormat.TRANSLUCENT;
        }
    }
}
