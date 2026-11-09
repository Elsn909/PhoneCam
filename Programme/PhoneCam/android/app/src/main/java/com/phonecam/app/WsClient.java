package com.phonecam.app;

import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

/**
 * Client WebSocket minimal (RFC 6455), même protocole que la page web de PhoneCam.
 * En Wi-Fi, TLS avec le certificat du PC épinglé : seule son empreinte (reçue par le lien
 * d'appairage) est acceptée, donc pas d'avertissement et pas d'interception possible.
 */
final class WsClient {
    interface Listener {
        void onText(String text);

        void onBinary(byte[] data);

        void onClosed(String reason);
    }

    private final Socket socket;
    private final OutputStream out;
    private final InputStream in;
    private final SecureRandom rnd = new SecureRandom();
    private final Listener listener;
    private volatile boolean closed;

    private WsClient(Socket s, Listener l) throws IOException {
        socket = s;
        out = s.getOutputStream();
        in = s.getInputStream();
        listener = l;
    }

    static WsClient connect(String hostPort, boolean tls, String token, String pin, Listener l) throws Exception {
        int colon = hostPort.lastIndexOf(':');
        String host = hostPort.substring(0, colon);
        int port = Integer.parseInt(hostPort.substring(colon + 1));
        Socket s;
        if (tls) {
            SSLContext ctx = SSLContext.getInstance("TLS");
            ctx.init(null, new TrustManager[]{new PinnedTrust(pin)}, null);
            SSLSocket ss = (SSLSocket) ctx.getSocketFactory().createSocket();
            ss.connect(new InetSocketAddress(host, port), 4000);
            ss.startHandshake();
            s = ss;
        } else {
            s = new Socket();
            s.connect(new InetSocketAddress(host, port), 4000);
        }
        s.setTcpNoDelay(true);
        s.setSoTimeout(0);
        WsClient ws = new WsClient(s, l);
        try {
            ws.handshake(hostPort, token);
        } catch (Exception e) {
            s.close();
            throw e;
        }
        Thread t = new Thread(ws::readLoop, "ws-lecture");
        t.setDaemon(true);
        t.start();
        return ws;
    }

    private void handshake(String hostPort, String token) throws IOException {
        byte[] nonce = new byte[16];
        rnd.nextBytes(nonce);
        String key = Base64.encodeToString(nonce, Base64.NO_WRAP);
        String req = "GET /ws?k=" + URLEncoder.encode(token, "UTF-8") + " HTTP/1.1\r\n"
                + "Host: " + hostPort + "\r\n"
                + "Upgrade: websocket\r\n"
                + "Connection: Upgrade\r\n"
                + "Sec-WebSocket-Key: " + key + "\r\n"
                + "Sec-WebSocket-Version: 13\r\n\r\n";
        out.write(req.getBytes(StandardCharsets.US_ASCII));
        out.flush();
        // En-têtes de la réponse, jusqu'à la ligne vide.
        ByteArrayOutputStream head = new ByteArrayOutputStream();
        int last4 = 0;
        while (true) {
            int b = in.read();
            if (b < 0) throw new EOFException("connexion fermée par le PC");
            head.write(b);
            last4 = (last4 << 8) | b;
            if (last4 == 0x0D0A0D0A) break;
            if (head.size() > 16384) throw new IOException("réponse trop longue");
        }
        String status = head.toString("US-ASCII").split("\r\n", 2)[0];
        if (!status.contains(" 101")) {
            throw new IOException(status.contains(" 403") ? "lien refusé par le PC (jeton périmé) : reconnecte-toi depuis le PC" : status);
        }
    }

    /** Envoie un message binaire (opcode 2) ou texte (opcode 1). Thread-safe. */
    synchronized void send(byte[] payload, int opcode) throws IOException {
        int n = payload.length;
        int hl = n < 126 ? 2 : n <= 0xFFFF ? 4 : 10;
        byte[] buf = new byte[hl + 4 + n];
        buf[0] = (byte) (0x80 | opcode);
        if (n < 126) {
            buf[1] = (byte) (0x80 | n);
        } else if (n <= 0xFFFF) {
            buf[1] = (byte) (0x80 | 126);
            buf[2] = (byte) (n >> 8);
            buf[3] = (byte) n;
        } else {
            buf[1] = (byte) (0x80 | 127);
            for (int i = 0; i < 8; i++) buf[2 + i] = (byte) ((long) n >> (56 - 8 * i));
        }
        byte[] mask = new byte[4];
        rnd.nextBytes(mask);
        System.arraycopy(mask, 0, buf, hl, 4);
        int o = hl + 4;
        for (int i = 0; i < n; i++) buf[o + i] = (byte) (payload[i] ^ mask[i & 3]);
        out.write(buf);
        out.flush();
    }

    void sendText(String s) throws IOException {
        send(s.getBytes(StandardCharsets.UTF_8), 1);
    }

    void close() {
        closed = true;
        try {
            socket.close();
        } catch (IOException ignored) {
        }
    }

    boolean isOpen() {
        return !closed;
    }

    private void readLoop() {
        String reason = "connexion perdue";
        try {
            while (!closed) {
                int b0 = readByte(), b1 = readByte();
                int op = b0 & 0x0F;
                long len = b1 & 0x7F;
                if (len == 126) len = (readByte() << 8) | readByte();
                else if (len == 127) {
                    len = 0;
                    for (int i = 0; i < 8; i++) len = (len << 8) | readByte();
                }
                byte[] mask = null;
                if ((b1 & 0x80) != 0) {
                    mask = new byte[4];
                    for (int i = 0; i < 4; i++) mask[i] = (byte) readByte();
                }
                if (len > 16 << 20) throw new IOException("message trop grand");
                byte[] data = new byte[(int) len];
                int got = 0;
                while (got < len) {
                    int r = in.read(data, got, (int) len - got);
                    if (r < 0) throw new EOFException();
                    got += r;
                }
                if (mask != null) for (int i = 0; i < data.length; i++) data[i] ^= mask[i & 3];
                if (op == 1) listener.onText(new String(data, StandardCharsets.UTF_8));
                else if (op == 2) listener.onBinary(data);                // écran / vidéos de Snapchat Web
                else if (op == 9) send(data, 10);               // ping → pong
                else if (op == 8) { reason = "fermée par le PC"; break; }
            }
        } catch (IOException e) {
            if (!closed) reason = "connexion perdue (" + e.getMessage() + ")";
        }
        boolean wasOpen = !closed;
        close();
        if (wasOpen) listener.onClosed(reason);
    }

    private int readByte() throws IOException {
        int b = in.read();
        if (b < 0) throw new EOFException("fin du flux");
        return b;
    }

    /** N'accepte que le certificat dont l'empreinte SHA-256 correspond au lien d'appairage. */
    private static final class PinnedTrust implements X509TrustManager {
        private final String pin;

        PinnedTrust(String pin) {
            this.pin = pin == null ? "" : pin;
        }

        @Override
        public void checkClientTrusted(X509Certificate[] chain, String authType) throws CertificateException {
            throw new CertificateException("inutilisé");
        }

        @Override
        public void checkServerTrusted(X509Certificate[] chain, String authType) throws CertificateException {
            if (chain == null || chain.length == 0 || pin.isEmpty()) throw new CertificateException("certificat absent");
            try {
                byte[] d = MessageDigest.getInstance("SHA-256").digest(chain[0].getEncoded());
                StringBuilder sb = new StringBuilder();
                for (byte x : d) sb.append(String.format("%02x", x));
                if (!sb.toString().equals(pin)) throw new CertificateException("ce n'est pas le PC appairé");
            } catch (java.security.NoSuchAlgorithmException e) {
                throw new CertificateException(e);
            }
        }

        @Override
        public X509Certificate[] getAcceptedIssuers() {
            return new X509Certificate[0];
        }
    }
}
