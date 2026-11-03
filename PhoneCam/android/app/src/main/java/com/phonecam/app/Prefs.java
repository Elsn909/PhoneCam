package com.phonecam.app;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;

/** Réglages mémorisés : lien vers le PC (reçu par le lien d'appairage) et options d'envoi. */
final class Prefs {
    String host = "";          // ip:port, ou localhost:port par le câble USB
    boolean tls = true;        // Wi-Fi : HTTPS avec certificat épinglé ; câble : en clair (local)
    String token = "";
    String pin = "";           // empreinte SHA-256 du certificat du PC
    boolean camera = true;
    boolean front = true;
    int width = 1280, height = 720, fps = 30, bitrate = 6_000_000;
    boolean mic = true;
    int micMode = AudioSource.MODE_SPEAKER;   // voir AudioSource.MODE_* (haut-parleur : capte plus loin)
    int micGain = 0;                          // gain ajouté, dB
    boolean eco = false;                      // économie : 24 img/s, débit réduit, Wi-Fi en veille possible

    private static SharedPreferences sp(Context c) {
        return c.getSharedPreferences("phonecam", Context.MODE_PRIVATE);
    }

    static Prefs load(Context c) {
        SharedPreferences s = sp(c);
        Prefs p = new Prefs();
        p.host = s.getString("host", p.host);
        p.tls = s.getBoolean("tls", p.tls);
        p.token = s.getString("token", p.token);
        p.pin = s.getString("pin", p.pin);
        p.camera = s.getBoolean("camera", p.camera);
        p.front = s.getBoolean("front", p.front);
        p.width = s.getInt("width", p.width);
        p.height = s.getInt("height", p.height);
        p.fps = s.getInt("fps", p.fps);
        p.bitrate = s.getInt("bitrate", p.bitrate);
        p.mic = s.getBoolean("mic", p.mic);
        p.micMode = s.getInt("micMode", p.micMode);
        p.mic = false;                            // le micro est celui du PC
        p.micGain = s.getInt("micGain", p.micGain);
        p.eco = s.getBoolean("eco", p.eco);
        return p;
    }

    void save(Context c) {
        sp(c).edit()
                .putString("host", host).putBoolean("tls", tls).putString("token", token).putString("pin", pin)
                .putBoolean("camera", camera).putBoolean("front", front)
                .putInt("width", width).putInt("height", height).putInt("fps", fps).putInt("bitrate", bitrate)
                .putBoolean("mic", mic).putInt("micMode", micMode).putInt("micGain", micGain).putBoolean("eco", eco)
                .apply();
    }

    /** phonecam://connect?h=hôte:port&s=1&k=jeton&fp=empreinte → vrai si le lien est valide. */
    boolean applyLink(Uri u) {
        if (u == null || !"phonecam".equals(u.getScheme())) return false;
        String h = u.getQueryParameter("h"), k = u.getQueryParameter("k");
        if (h == null || k == null) return false;
        host = h;
        token = k;
        tls = "1".equals(u.getQueryParameter("s"));
        String fp = u.getQueryParameter("fp");
        pin = fp == null ? "" : fp.toLowerCase();
        return true;
    }

    boolean linked() {
        return !host.isEmpty() && !token.isEmpty();
    }
}
