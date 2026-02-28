package com.trackit.myapplication
import android.provider.Settings
import kotlin.math.max
import android.Manifest
import android.annotation.SuppressLint
import android.app.*
import android.content.*
import android.graphics.*
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.Uri
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.net.wifi.WifiNetworkSpecifier
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkRequest
import android.os.*
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.Window
import android.view.WindowManager
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.util.*
import kotlin.collections.HashMap
import kotlin.concurrent.thread
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
// ---------------- NODE & ORCHESTRATOR ----------------
import okhttp3.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

// -------------------------------
// IMPORTANT: Manifest notes (ADD THESE)
// -------------------------------
// <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
// <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION"/>
// <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
// <uses-permission android:name="android.permission.INTERNET"/>
// <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
// <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
// <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE"/>
//
// Inside <application> add:
// <service android:name=".MainActivity$ForegroundWorker" android:exported="false" android:foregroundServiceType="location" />
//
// (If targeting Android 13+, handle new media permissions accordingly.)
// -------------------------------

/**
 * Merged MainActivity.kt
 *
 * - Your original app code (detection grid, UDP listeners, image reassembly, GPS)
 * - Added: Pi AP scanning + connect popup
 * - Added: Best-route UI and routing engine
 * - Added: Offline recovery scanning
 * - Added: Supabase toggle (send logs/metadata when enabled and key present)
 * - Added: PortalActivity (WebView) for portal render/auth flow
 *
 * Keep things minimal and non-destructive.
 */

class MainActivity : AppCompatActivity() {
    // =================================================================================
// ====================== CORE DATA STRUCTURES & REGISTRIES ========================
// =================================================================================

    data class NodeInfo(
        val id: String,
        val ip: String,
        val role: String,
        var load: Double,
        var internet: Boolean,
        var lastSeen: Long = System.currentTimeMillis()
    )

    data class BBox(
        val x1: Int,
        val y1: Int,
        val x2: Int,
        val y2: Int,
        val cls: String,
        val trackId: Int
    )

    data class Detection(
        val filename: String,
        val className: String,
        val trackId: Int,
        val ts: Long,
        val lat: Double?,
        val lon: Double?,
        val rawBytes: ByteArray,
        val bitmap: Bitmap,
        val boxes: MutableList<BBox> = mutableListOf()
    )

    class ImageBuffer(val totalChunks: Int) {
        val chunks = mutableMapOf<Int, ByteArray>()
    }

    private val nodeRegistry = Collections.synchronizedMap(HashMap<String, NodeInfo>())
    private val detections = Collections.synchronizedList(mutableListOf<Detection>())

    // =================================================================================
// ====================== NETWORK CONSTANTS ========================================
// =================================================================================

    private const val VIDEO_PORT = 5001
    private const val LOG_PORT = 5002
    private const val META_PORT = 5003
    private const val IMAGE_PORT = 5004
    private const val COMMAND_PORT = 5005
    private const val HEARTBEAT_PORT = 5006

    private const val SECRET = "memoryretrieve##$"


    private const val PREF_SUPABASE_URL = "supabase_url"
    // ----------------- Constants / Ports / Secret -----------------
    private val NODE_HEARTBEAT_PORT = 7777

    private val DISCOVERY_PORT = 5004


    // ----------------- Data models -----------------
    data class GPSEntry(val ts: Long, val lat: Double, val lon: Double)

    class ImageAssembler(private val total: Int) {
        private val received = BooleanArray(total)
        private val buffers = arrayOfNulls<ByteArray>(total)
        private var receivedCount = 0

        fun insert(seq: Int, data: ByteArray) {
            if (seq < 0 || seq >= total) return
            if (received[seq]) return

            buffers[seq] = data
            received[seq] = true
            receivedCount++
        }

        fun isComplete(): Boolean = receivedCount == total

        fun build(): ByteArray {
            val size = buffers.sumOf { it?.size ?: 0 }
            val out = ByteArray(size)
            var pos = 0
            for (i in 0 until total) {
                val b = buffers[i] ?: continue
                System.arraycopy(b, 0, out, pos, b.size)
                pos += b.size
            }
            return out
        }
    }
    data class NodeInfo(val id: String, val role: String, val ip: String, val lastSeen: Long, val internet: Boolean, val load: Double)

    // ----------------- UI elements -----------------
    private lateinit var imgLive: ImageView
    private lateinit var grid: GridView
    private lateinit var etSearch: EditText
    private lateinit var btnSearch: Button
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var tvStatus: TextView

    // Orchestration UI elements created programmatically
    private var routePanel: LinearLayout? = null
    private var supabaseSwitch: Switch? = null

    // ----------------- Local state & handler -----------------
    private val handler = Handler(Looper.getMainLooper())
    private val filtered = mutableListOf<Detection>()
    private val gpsLog = Collections.synchronizedList(mutableListOf<GPSEntry>())

    private lateinit var locationManager: LocationManager

    // Foreground worker instance (owns all background threads & sockets)
    private var worker: ForegroundWorker? = null

    // manual override route key
    private var manualOverride: String? = null

    // shared pref key for supabase enable + key
    private val PREFS = "memoryretrieve_prefs"
    private val PREF_SUPABASE_ENABLED = "supabase_enabled"
    private val PREF_SUPABASE_KEY = "supabase_key" // put your key here securely if you want uploads
    
    class MemoryRetrieveService : Service() {
        private lateinit var worker: MainActivity.ForegroundWorker

        override fun onCreate() {
            super.onCreate()
            worker = MainActivity.ForegroundWorker(this)
            worker.start() // Start worker threads
        }

        override fun onDestroy() {
            super.onDestroy()
            worker.stopWorker()
        }

        override fun onBind(intent: Intent?): IBinder? = null
    }
    // --------------------------------------------------------------------------------
    // Activity lifecycle / UI wiring
    // --------------------------------------------------------------------------------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgLive = findViewById(R.id.imgLive)
        grid = findViewById(R.id.gridDetections)
        etSearch = findViewById(R.id.etSearch)
        btnSearch = findViewById(R.id.btnSearch)
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        tvStatus = findViewById(R.id.tvStatus)

        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager
        // request permissions (fine location + storage). App should ask user and handle denials.
        requestRuntimePermissions()

        // Adapter (keeps your original view logic)
        grid.adapter = object : BaseAdapter() {
            override fun getCount() = filtered.size
            override fun getItem(p: Int) = filtered[p]
            override fun getItemId(p: Int) = p.toLong()
            override fun getView(p: Int, v: android.view.View?, parent: android.view.ViewGroup?): android.view.View {
                val view = v ?: layoutInflater.inflate(R.layout.item_detection, parent, false)
                val d = filtered[p]

                val baseBmp = d.bitmap ?: try {
                    BitmapFactory.decodeByteArray(d.imageBytes, 0, d.imageBytes.size)
                } catch (e: Exception) { null }

                val mutable = baseBmp?.copy(Bitmap.Config.ARGB_8888, true) ?: Bitmap.createBitmap(1,1,Bitmap.Config.ARGB_8888)
                val canvas = Canvas(mutable)

                val paint = Paint().apply {
                    color = Color.GREEN
                    style = Paint.Style.STROKE
                    strokeWidth = 4f
                    textSize = 28f
                }

                for (b in d.boxes) {
                    canvas.drawRect(b.x1.toFloat(), b.y1.toFloat(), b.x2.toFloat(), b.y2.toFloat(), paint)
                    canvas.drawText("${b.label}_${b.trackId}", b.x1.toFloat() + 4, b.y1.toFloat() - 6, paint)
                }

                view.findViewById<ImageView>(R.id.imgThumb).setImageBitmap(mutable)
                view.findViewById<TextView>(R.id.textClass).text = d.className.uppercase()
                view.findViewById<TextView>(R.id.textTrack).text = "ID: ${d.trackId}"

                view.setOnClickListener {
                    d.lat?.let { lat ->
                        d.lon?.let { lon ->
                            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=$lat,$lon(${d.className})")))
                        }
                    }
                }

                return view
            }
        }

        btnStart.setOnClickListener { worker?.sendCommandToPi("START") }
        btnStop.setOnClickListener { worker?.sendCommandToPi("STOP") }
        btnSearch.setOnClickListener { filterDetections() }

        // Start the foreground worker (keeps running while screen off)
        startForegroundService(Intent(this, MemoryRetrieveService::class.java))

        // build orchestration UI overlay (route panel + supabase switch)
        setupOrchestrationUI()
        startNetworkOrchestration()
        startOrchestratorBrain()


        // Periodic UI refresh from worker state
        handler.post(uiRefresh)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(uiRefresh)
        worker?.stopWorker()
        worker = null
    }

    private val uiRefresh = object : Runnable {
        override fun run() {
            try {
                worker?.let { w ->
                    // update live image if exists
                    w.latestVideoFrame?.let { bmp ->
                        imgLive.setImageBitmap(bmp)
                    }

                    // copy detections from worker
                    synchronized(detections) {
                        detections.clear()
                        detections.addAll(w.detections.map { det ->
                            Detection(det.filename, det.className, det.trackId, det.ts, det.lat, det.lon, det.imageBytes, det.bitmap, Collections.synchronizedList(det.boxes.toMutableList()))
                        })
                    }
                    tvStatus.text = "PI: ${w.piIp ?: "not found"}"
                }
                filterDetections()

                // update orchestration UI (routes)
                updateRoutePanel()

            } catch (e: Exception) {
                Log.e("MA-UI", "refresh error: ${e.message}")
            } finally {
                handler.postDelayed(this, 800)
            }
        }
    }

    private fun filterDetections() {
        val query = etSearch.text.toString().lowercase()
        filtered.clear()
        filtered.addAll(if (query.isBlank()) detections else detections.filter { it.className.lowercase().contains(query) })
        (grid.adapter as BaseAdapter).notifyDataSetChanged()
    }

    // --------------------------------------------------------------------------------
    // Orchestration UI: route panel + supabase toggle
    // --------------------------------------------------------------------------------
    private fun setupOrchestrationUI() {
        // route panel (bottom overlay)
        routePanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            elevation = 10f
            setPadding(12,12,12,12)
        }

        // supabase switch row
        val supRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val supLabel = TextView(this).apply {
            text = "Upload logs to Supabase"
            textSize = 14f
        }
        supabaseSwitch = Switch(this)
        // load saved pref
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        supabaseSwitch?.isChecked = prefs.getBoolean(PREF_SUPABASE_ENABLED, false)
        supabaseSwitch?.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean(PREF_SUPABASE_ENABLED, isChecked).apply()
            showToast("Supabase uploads: ${if (isChecked) "ENABLED" else "DISABLED"}")
        }
        supRow.addView(supLabel, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        supRow.addView(supabaseSwitch, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))

        routePanel?.addView(supRow)

        // add the routePanel as overlay bottom
        val params = FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM)
        addContentView(routePanel, params)
    }

    // route computing & UI
    data class RouteOption(val key: String, val label: String, val score: Int)

    private fun computeRoutes(nodes: List<NodeInfo>): List<RouteOption> {
        val sorted = nodes.sortedByDescending { computeNodeScore(it) }

        val roles = sorted.map { it.role.uppercase() }.toSet()

        val hasZero = roles.contains("PI_ZERO")
        val hasPi5 = roles.contains("PI5")
        val hasLaptop = roles.contains("LAPTOP")

        val list = mutableListOf<RouteOption>()

        if (hasZero && hasLaptop) list.add(RouteOption("Z_P_L", "PI ZERO + PHONE + LAPTOP", 100))
        if (hasLaptop && hasPi5) list.add(RouteOption("P_5_L", "PHONE + PI5 + LAPTOP", 90))
        if (hasZero && hasPi5) list.add(RouteOption("Z_P_5", "PI ZERO + PHONE + PI5", 85))
        if (hasZero) list.add(RouteOption("Z_P", "PI ZERO + PHONE", 70))
        if (hasPi5) list.add(RouteOption("P_5", "PHONE + PI5", 65))

        if (list.isEmpty()) list.add(RouteOption("PHONE_ONLY", "PHONE ONLY (fallback)", 10))

        return list
    }
    enum class SystemMode {
        PHONE_ONLY,
        MINIMAL_FIELD,     // Phone + Pi Zero
        EDGE_COMPUTE,      // Phone + Pi Zero + Pi5
        AI_CORE,           // Phone + Pi Zero + Pi5 + Laptop
        EMERGENCY_AI       // Phone + Laptop
    }
    data class ActiveRoute(
        val sensor: String,
        val compute: String,
        val ai: String
    )
    private fun determineSystemMode(): SystemMode {
        val roles = nodes.values.map { it.role.uppercase() }.toSet()

        return when {
            roles.isEmpty() -> SystemMode.PHONE_ONLY
            roles.contains("PI_ZERO") && roles.contains("PI5") && roles.contains("LAPTOP") -> SystemMode.AI_CORE
            roles.contains("PI_ZERO") && roles.contains("PI5") -> SystemMode.EDGE_COMPUTE
            roles.contains("PI_ZERO") -> SystemMode.MINIMAL_FIELD
            roles.contains("LAPTOP") -> SystemMode.EMERGENCY_AI
            else -> SystemMode.PHONE_ONLY
        }
    }
    private fun computeOptimalRoute(mode: SystemMode): ActiveRoute {
        val zero = nodes.values.find { it.role.uppercase() == "PI_ZERO" }?.id
        val pi5  = nodes.values.find { it.role.uppercase() == "PI5" }?.id
        val lap  = nodes.values.find { it.role.uppercase() == "LAPTOP" }?.id

        return when (mode) {
            SystemMode.PHONE_ONLY -> ActiveRoute("PHONE", "PHONE", "PHONE")
            SystemMode.MINIMAL_FIELD -> ActiveRoute(zero ?: "PHONE", "PHONE", "PHONE")
            SystemMode.EDGE_COMPUTE -> ActiveRoute(zero ?: "PHONE", pi5 ?: "PHONE", "PHONE")
            SystemMode.AI_CORE -> ActiveRoute(zero ?: "PHONE", pi5 ?: "PHONE", lap ?: "PHONE")
            SystemMode.EMERGENCY_AI -> ActiveRoute("PHONE", lap ?: "PHONE", lap ?: "PHONE")
        }
    }
    @Volatile private var lastRoute: ActiveRoute? = null

    private fun enforceRoute(route: ActiveRoute) {
        if (route == lastRoute) return
        lastRoute = route

        Log.d("ORCH", "Applying route → $route")

        worker?.sendCommandToAllNodes("STANDBY")

        fun activate(target: String, cmd: String) {
            if (target == "PHONE") return
            nodes[target]?.let { sendCommandToNode(it, cmd) }
        }

        activate(route.sensor, "START_CAMERA")
        activate(route.compute, "START_COMPUTE")
        activate(route.ai, "START_AI")
    }
    private fun startOrchestratorBrain() {
        CoroutineScope(Dispatchers.IO).launch {
            while (true) {
                try {
                    val mode = determineSystemMode()
                    val route = computeOptimalRoute(mode)
                    enforceRoute(route)
                } catch (e: Exception) {
                    Log.e("ORCH", "brain error: ${e.message}")
                }
                delay(3000)
            }
        }
    }
    // ---------------- NODE MANAGER ----------------
    data class Node(
        val id: String,
        val role: String,
        val ip: String,
        var load: Double,
        var internet: Boolean,
        var lastSeen: Long
    )

    val nodes = ConcurrentHashMap<String, Node>()
    val heartbeatPort = 5005
    val udpSocket = DatagramSocket()

    fun startNodeHeartbeatListener() {
        CoroutineScope(Dispatchers.IO).launch {
            val buf = ByteArray(2048)
            while (true) {
                try {
                    val packet = DatagramPacket(buf, buf.size)
                    udpSocket.receive(packet)
                    val data = String(packet.data, 0, packet.length)
                    if (!data.contains("|")) continue
                    val parts = data.split("|", limit = 2)
                    val signature = parts[0]
                    val payloadJson = parts[1]
                    // TODO: verify signature
                    val info = parseNodeJson(payloadJson)
                    nodes[info.id] = info.copy(lastSeen = System.currentTimeMillis())
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    fun parseNodeJson(json: String): Node {
        val map = mutableMapOf<String, String>()
        json.trim('{', '}').split(",").forEach {
            val kv = it.split(":")
            if (kv.size == 2) map[kv[0].trim('"')] = kv[1].trim('"')
        }
        return Node(
            id = map["id"] ?: "unknown",
            role = map["role"] ?: "UNKNOWN",
            ip = map["ip"] ?: "0.0.0.0",
            load = map["load"]?.toDoubleOrNull() ?: 999.0,
            internet = map["internet"]?.toBoolean() ?: false,
            lastSeen = System.currentTimeMillis()
        )
    }

    fun getBestNode(): Node? {
        val now = System.currentTimeMillis()
        return nodes.values.filter { now - it.lastSeen < 15_000 }
            .maxByOrNull { computeNodeScore(it) }
    }

    fun computeNodeScore(node: Node): Int {
        var score = 0
        score += if (node.internet) 30 else 0
        score += if (node.role.uppercase() == "LAPTOP") 50 else 0
        score += (10 - minOf(node.load, 10)).toInt()
        return score
    }

    // ---------------- COMMAND DISPATCHER ----------------
    fun sendCommandToNode(node: Node, command: String, params: Map<String, String> = emptyMap()) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val payload = mapOf(
                    "command" to command,
                    "params" to params
                )
                val jsonPayload = JSONObject(payload).toString()
                val packetData = jsonPayload.toByteArray()
                val sock = DatagramSocket()
                sock.send(
                    DatagramPacket(packetData, packetData.size, InetAddress.getByName(node.ip), 5006)
                )
                sock.close()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    // ---------------- PORTAL ENGINE ----------------
    val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()

    fun replayPortalSubmission(
        node: Node,
        actionUrl: String,
        method: String,
        formData: Map<String, String>
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val request = if (method.uppercase() == "POST") {

                    val bodyBuilder = FormBody.Builder()
                    formData.forEach { (k, v) -> bodyBuilder.add(k, v) }

                    Request.Builder()
                        .url(actionUrl)
                        .post(bodyBuilder.build())
                        .header("User-Agent", "Mozilla/5.0")
                        .header("Referer", actionUrl)
                        .build()

                } else {

                    val httpUrl = actionUrl.toHttpUrlOrNull()?.newBuilder() ?: return@launch
                    formData.forEach { (k, v) -> httpUrl.addQueryParameter(k, v) }

                    Request.Builder()
                        .url(httpUrl.build())
                        .get()
                        .header("User-Agent", "Mozilla/5.0")
                        .header("Referer", actionUrl)
                        .build()
                }

                val response = client.newCall(request).execute()

                try {
                    if (response.isSuccessful) {
                        Log.d("PORTAL", "Submission success → ${node.id}")
                        sendCommandToNode(node, "portal_success")
                    } else {
                        Log.w("PORTAL", "Submission failed ${response.code}")
                    }
                } finally {
                    response.close()
                }

            } catch (e: Exception) {
                Log.e("PORTAL", "Submission crash: ${e.message}", e)
            }
        }
    }

    // ---------------- PERIODIC NODE POLL ----------------
    fun startNodePoller() {
        CoroutineScope(Dispatchers.IO).launch {
            while (true) {
                val best = getBestNode()
                best?.let {
                    println("[NODE] Best node: ${it.id} (${it.role}) score=${computeNodeScore(it)}")
                }
                delay(5000)
            }
        }
    }
    private fun openWifiPanel() {
        startActivity(Intent(Settings.Panel.ACTION_WIFI))
    }
    private var piPopupVisible = false

    private fun showPiPopup(ssid: String) {
        if (piPopupVisible) return
        piPopupVisible = true

        val dialog = Dialog(this, android.R.style.Theme_Translucent_NoTitleBar)
        dialog.setContentView(R.layout.popup_pi_connect)
        dialog.window?.setGravity(Gravity.BOTTOM)

        dialog.findViewById<Button>(R.id.btnConnect).setOnClickListener {
            openWifiPanel()
            dialog.dismiss()
            piPopupVisible = false
        }

        dialog.show()
    }
    @SuppressLint("MissingPermission")
    private fun scanForPiAp() {
        val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val results = try { wifiManager.scanResults } catch (_: Exception) { emptyList<ScanResult>() }

        val pi = results.firstOrNull {
            val s = it.SSID ?: ""
            s.contains("PI", true) || s.contains("ZERO", true)
        }

        if (pi != null) {
            runOnUiThread {
                showPiPopup(pi.SSID ?: "PI")
            }
        }
    }
    private fun startNetworkOrchestration() {
        Thread {
            while (true) {
                try {
                    scanForPiAp()
                } catch (_: Exception) {}
                Thread.sleep(3000)
            }
        }.start()
    }
    private fun updateRoutePanel() {
        try {
            val panel = routePanel ?: return
            val nodes = worker?.nodeRegistry?.values?.toList() ?: emptyList()
            val routes = computeRoutes(nodes)

            // clear older route views but keep the supabase switch row (index 0)
            panel.removeViews(1, max(0, panel.childCount - 1))

            // insert title
            val title = TextView(this).apply {
                text = "SYSTEM ROUTE"
                textSize = 16f
                setTypeface(null, Typeface.BOLD)
                setPadding(6,6,6,6)
            }
            panel.addView(title)

            routes.forEachIndexed { i, r ->
                val isBest = i == 0
                val isActive = (manualOverride != null && manualOverride == r.key)

                val tv = TextView(this).apply {
                    textSize = if (isBest) 15f else 13f
                    setPadding(12,12,12,12)
                    text = buildString {
                        if (isBest) append("⭐ BEST  ")
                        if (isActive) append("🟢 ACTIVE  ")
                        append(r.label)
                    }
                    setBackgroundColor(if (isActive) Color.parseColor("#C8F7C5") else if (isBest) Color.parseColor("#FFF8C4") else Color.parseColor("#EEEEEE"))
                    setOnClickListener {
                        manualOverride = r.key
                        applyRoute(r.key)
                        showToast("Manual route set: ${r.label}")
                    }
                }
                panel.addView(tv)
            }

            // small spacer
            val spacer = Space(this)
            spacer.minimumHeight = 6
            panel.addView(spacer)

        } catch (e: Exception) {
            Log.e("ROUTE-UI", "update error: ${e.message}")
        }
    }

    // Route enforcement: send simple commands to nodes via worker.
    private fun applyRoute(key: String) {
        when (key) {
            "Z_P_L" -> {
                // Pi Zero streams, laptop computes
                worker?.sendCommandToAllNodes("STREAM=PI_ZERO")
                worker?.sendCommandToAllNodes("COMPUTE=LAPTOP")
            }
            "Z_P_5" -> {
                worker?.sendCommandToAllNodes("STREAM=PI_ZERO")
                worker?.sendCommandToAllNodes("COMPUTE=PI5")
            }
            "P_5_L" -> {
                worker?.sendCommandToAllNodes("STREAM=PI5")
                worker?.sendCommandToAllNodes("COMPUTE=LAPTOP")
            }
            "Z_P" -> {
                worker?.sendCommandToAllNodes("STREAM=PI_ZERO")
                worker?.sendCommandToAllNodes("COMPUTE=PHONE")
            }
            "P_5" -> {
                worker?.sendCommandToAllNodes("STREAM=PI5")
                worker?.sendCommandToAllNodes("COMPUTE=PHONE")
            }
            "PHONE_ONLY" -> {
                worker?.sendCommandToAllNodes("STREAM=NONE")
                worker?.sendCommandToAllNodes("COMPUTE=PHONE")
            }
        }
    }

    // helper: show toast
    private fun showToast(msg: String) {
        runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
    }

    // ----------------- ForegroundWorker: runs in-process background threads and holds all sockets/state
    // (keeps everything in single file per user's request)
    // --------------------------------------------------------------------------------
    private val videoFrameLock = Any()
    inner class ForegroundWorker(private val ctx: Context) {
        // Shared state exposed to Activity
        val detections = Collections.synchronizedList(mutableListOf<Detection>())
        val nodeRegistry = Collections.synchronizedMap(HashMap<String, NodeInfo>())
        val gpsLog = Collections.synchronizedList(mutableListOf<GPSEntry>())

        @Volatile var latestVideoFrame: Bitmap? = null
        @Volatile var piIp: String? = null

        // internal
        private var running = true
        private var job: Job? = null
        private var notifId = 2345

        // Local location listener used by worker
        private var locationListener: LocationListener? = null

        // For portal forwarding
        @Volatile private var lastPortalHtml: String? = null
        @Volatile private var lastPortalFromIp: String? = null

        // Supabase upload control => read settings from MainActivity via SharedPreferences in worker methods
        private val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)

        private fun probeLatency(ip: String): Long {
            return try {
                val start = System.currentTimeMillis()
                InetAddress.getByName(ip).isReachable(150)
                System.currentTimeMillis() - start
            } catch (e: Exception) {
                9999L
            }
        }
        fun start() {
            // Start foreground notification to keep process alive when screen off
            startForegroundCompat()

            // Kick off threads
            startGpsLogging()
            startSwarmHeartbeatListener()
            startDiscovery()
            startUdpListeners()
            startImageListener()
            startMetadataListener()

            // NEW: network recovery + Pi AP watcher
            startPiApWatcher()
            networkBrainLoop()

            // NEW: periodic supabase flush (if enabled)
            startSupabaseFlushLoop()
        }

        fun stopWorker() {
            running = false
            try {
                locationListener?.let { locationManager.removeUpdates(it) }
            } catch (_: Exception) {}
            job?.cancel()
        }

        // ----------------- Foreground notification (keeps it alive) -----------------
// Call this inside a true Service (MemoryRetrieveService)
        private fun startForegroundCompat(service: Service) {
            try {
                val nm = service.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                val channelId = "memretr_service_chan"

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val ch = NotificationChannel(
                        channelId,
                        "MemoryRetrieve",
                        NotificationManager.IMPORTANCE_LOW
                    )
                    nm.createNotificationChannel(ch)
                }

                val intent = Intent(service, MainActivity::class.java)
                val pendingFlags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
                    PendingIntent.FLAG_IMMUTABLE
                else 0
                val pIntent = PendingIntent.getActivity(service, 0, intent, pendingFlags)

                val notif = NotificationCompat.Builder(service, channelId)
                    .setContentTitle("MemoryRetrieve running")
                    .setContentText("Background services active")
                    .setSmallIcon(R.drawable.ic_launcher_foreground)
                    .setContentIntent(pIntent)
                    .setOngoing(true)
                    .build()

                // Start true foreground
                service.startForeground(notifId, notif)

            } catch (e: Exception) {
                Log.w("FW-NOTIF", "startForegroundCompat error: ${e.message}")
            }
        }

        // ----------------- GPS Logging -----------------
        @SuppressLint("MissingPermission")
        private fun startGpsLogging() {
            thread {
                try {
                    locationListener = object : LocationListener {
                        override fun onLocationChanged(location: Location) {
                            try {
                                gpsLog.add(GPSEntry(System.currentTimeMillis(), location.latitude, location.longitude))
                                if (gpsLog.size > 1000) gpsLog.removeAt(0)
                            } catch (e: Exception) {
                                Log.e("FW-GPS", "listener error: ${e.message}")
                            }
                        }
                        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                        override fun onProviderEnabled(provider: String) {}
                        override fun onProviderDisabled(provider: String) {}
                    }

                    try {
                        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, locationListener!!, Looper.getMainLooper())
                    } catch (se: SecurityException) {
                        Log.w("FW-GPS", "no permission for requestLocationUpdates")
                    } catch (e: Exception) {
                        Log.w("FW-GPS", "requestLocationUpdates failed: ${e.message}")
                    }

                } catch (e: Exception) {
                    Log.e("FW-GPS", "start error: ${e.message}")
                }

                while (running) {
                    try {
                        val loc: Location? = try {
                            locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                                ?: locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                        } catch (se: SecurityException) {
                            null
                        }

                        loc?.let {
                            gpsLog.add(GPSEntry(System.currentTimeMillis(), it.latitude, it.longitude))
                            if (gpsLog.size > 1000) gpsLog.removeAt(0)
                        }
                    } catch (e: Exception) {
                        Log.e("FW-GPS", "poll error: ${e.message}")
                    }
                    Thread.sleep(1500)
                }
            }
        }

        fun getNearestGps(ts: Long): GPSEntry? {
            synchronized(gpsLog) {
                if (gpsLog.isEmpty()) return null
                return gpsLog.minByOrNull { kotlin.math.abs(it.ts - ts) }
            }
        }

        // ----------------- Swarm heartbeat listener -----------------
        private fun startSwarmHeartbeatListener() {
            thread {
                try {
                    DatagramSocket(NODE_HEARTBEAT_PORT).use { socket ->
                        val buf = ByteArray(4096)
                        while (running) {
                            try {
                                val packet = DatagramPacket(buf, buf.size)
                                socket.receive(packet)
                                val msg = String(packet.data, 0, packet.length)
                                val json = JSONObject(msg)
                                val node = NodeInfo(
                                    id = json.getString("id"),
                                    role = json.optString("role", "UNKNOWN"),
                                    ip = json.optString("ip", packet.address.hostAddress),
                                    lastSeen = System.currentTimeMillis(),
                                    internet = json.optBoolean("internet", false),
                                    load = json.optDouble("load", 99.0)
                                )
                                nodeRegistry[node.id] = node
                                // cleanup stale nodes
                                val now = System.currentTimeMillis()
                                val toRemove = nodeRegistry.keys.filter { nid ->
                                    now - (nodeRegistry[nid]?.lastSeen ?: 0L) > 11000L
                                }
                                toRemove.forEach { nodeRegistry.remove(it) }
                            } catch (e: Exception) {
                                Log.e("FW-SWARM", "heartbeat parse error: ${e.message}")
                                Thread.sleep(50)
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("FW-SWARM", "socket error: ${e.message}")
                }
            }
        }

        // ----------------- Discovery (broadcast) -----------------
        private fun startDiscovery() {
            thread {
                try {
                    DatagramSocket(DISCOVERY_PORT).use { socket ->
                        socket.broadcast = true
                        // reply listener
                        thread {
                            while (running) {
                                try {
                                    val buf = ByteArray(1024) // allocate per iteration
                                    val packet = DatagramPacket(buf, buf.size)
                                    socket.receive(packet)
                                    val msg = String(packet.data, 0, packet.length)
                                    if (msg.startsWith("OK|PI|")) {
                                        piIp = msg.substring(6)
                                        Log.d("FW-DISC", "Found PI: $piIp")
                                    }
                                } catch (e: Exception) {
                                    // ignore
                                }
                            }
                        }

                        // broadcast loop
                        while (running) {
                            try {
                                val broadcastAddr = InetAddress.getByName("255.255.255.255")
                                val packet = DatagramPacket(secretMsg, secretMsg.size, broadcastAddr, DISCOVERY_PORT)
                                socket.send(packet)
                            } catch (e: Exception) {
                                Log.e("FW-DISC", "broadcast error: ${e.message}")
                            }
                            Thread.sleep(3000)
                        }
                    }
                } catch (e: Exception) {
                    Log.e("FW-DISC", "discovery socket error: ${e.message}")
                }
            }
        }

        // ----------------- UDP listeners: video + logs -----------------
        private fun startUdpListeners() {
// ----------------- VIDEO LISTENER -----------------
            thread(name = "UDP-Video-Listener") {
                DatagramSocket(null).use { socket ->
                    try {
                        socket.reuseAddress = true
                        socket.bind(InetSocketAddress(VIDEO_PORT))
                        socket.soTimeout = 500 // allows clean exit

                        while (running) {
                            try {
                                val buf = ByteArray(65507)   // allocate per packet (NO reuse)
                                val packet = DatagramPacket(buf, buf.size)
                                socket.receive(packet)

                                val bmp = BitmapFactory.decodeByteArray(packet.data, 0, packet.length)
                                bmp?.let { newFrame ->
                                    synchronized(this) {
                                        // recycle old bitmap to prevent memory leak
                                        latestVideoFrame?.recycle()
                                        latestVideoFrame = newFrame
                                    }
                                }
                            } catch (e: SocketTimeoutException) {
                                // allows loop to exit cleanly when running=false
                            } catch (e: Exception) {
                                Log.e("FW-VIDEO", "video recv error: ${e.message}")
                                Thread.sleep(10)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("FW-VIDEO", "video socket fail: ${e.message}")
                    }
                }
            }

            // ----------------- LOG LISTENER -----------------
            thread(name = "UDP-Log-Listener") {
                DatagramSocket(null).use { socket ->
                    try {
                        socket.reuseAddress = true
                        socket.bind(InetSocketAddress(LOG_PORT))
                        socket.soTimeout = 500

                        while (running) {
                            val buf = ByteArray(16384) // allocate per packet
                            try {
                                val packet = DatagramPacket(buf, buf.size)
                                socket.receive(packet)
                                val msg = String(packet.data, 0, packet.length, Charsets.UTF_8)

                                // Handle portal JSON forwards
                                try {
                                    val json = JSONObject(msg)
                                    if (json.optString("type") == "PORTAL_PAGE") {
                                        val html = json.optString("html", "")
                                        lastPortalHtml = html
                                        lastPortalFromIp = packet.address.hostAddress

                                        // UI thread call using Activity context
                                        (runOnUiThread) {
                                            openPortalActivity(html, lastPortalFromIp)
                                        }
                                        continue
                                    }
                                } catch (_: Exception) {
                                    // not JSON, ignore
                                }

                                Log.d("PiLOG", "Received: $msg")
                            } catch (e: SocketTimeoutException) {
                                // allow loop to check `running`
                            } catch (e: Exception) {
                                Log.e("FW-LOG", "Log receive error: ${e.message}")
                                Thread.sleep(10)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("FW-LOG", "Log socket failed: ${e.message}")
                    }
                }
            }
        }

        // ----------------- Metadata listener (bbox messages) -----------------
        private fun startMetadataListener() {
            thread(name = "UDP-Meta-Listener") {
                DatagramSocket(null).use { socket ->
                    try {
                        socket.reuseAddress = true
                        socket.bind(InetSocketAddress(META_PORT))
                        socket.soTimeout = 500 // allow periodic running check
                        val buf = ByteArray(8192)

                        while (running) {
                            try {
                                val packet = DatagramPacket(buf, buf.size)
                                socket.receive(packet)
                                val msg = String(packet.data, 0, packet.length, Charsets.UTF_8)

                                val json = try { JSONObject(msg) } catch (e: Exception) {
                                    Log.w("FW-META", "Invalid JSON: ${e.message}")
                                    continue
                                }

                                if (json.optString("type") != "bbox") continue

                                val fname = json.getString("filename")
                                val bboxArray = json.getJSONArray("bbox")
                                val box = BBox(
                                    bboxArray.getInt(0),
                                    bboxArray.getInt(1),
                                    bboxArray.getInt(2),
                                    bboxArray.getInt(3),
                                    json.getString("classname"),
                                    json.getInt("trackId")
                                )

                                // Thread-safe addition to detection
                                synchronized(detections) {
                                    val found = detections.find { it.filename == fname }
                                    found?.boxes?.add(box)
                                }

                            } catch (e: SocketTimeoutException) {
                                // ignore timeout, allows loop to check running
                            } catch (e: Exception) {
                                Log.e("FW-META", "BBox receive error: ${e.message}")
                                Thread.sleep(10)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("FW-META", "Meta socket failed: ${e.message}")
                    }
                }
            }
        }

        // ----------------- Image chunk receiver -----------------
        private fun startImageListener() {
            thread(name = "UDP-Image-Listener") {
                try {
                    DatagramSocket(null).use { socket ->
                        socket.reuseAddress = true
                        socket.bind(InetSocketAddress(IMAGE_PORT))
                        socket.soTimeout = 500

                        val imageBuffers = HashMap<String, ImageAssembler>()

                        while (running) {
                            try {
                                val buf = ByteArray(65507)
                                val packet = DatagramPacket(buf, buf.size)
                                socket.receive(packet)

                                // find first 6 '|' delimiters
                                val pipeIdxs = ArrayList<Int>(6)
                                for (i in 0 until packet.length) {
                                    if (packet.data[i].toInt() == '|'.code) {
                                        pipeIdxs.add(i)
                                        if (pipeIdxs.size == 6) break
                                    }
                                }
                                if (pipeIdxs.size < 6) continue

                                val headerEnd = pipeIdxs[5] + 1
                                val headerStr = String(packet.data, 0, headerEnd, Charsets.UTF_8)
                                val parts = headerStr.split("|")
                                if (parts.size < 6 || parts[0] != "IMG") continue

                                val fname = parts[1]
                                val seq = parts[2].toIntOrNull() ?: continue
                                val total = parts[3].toIntOrNull() ?: continue
                                val payloadLen = parts[5].toIntOrNull() ?: continue

                                if (headerEnd + payloadLen > packet.length) continue

                                val assembler = imageBuffers.getOrPut(fname) {
                                    ImageAssembler(total)
                                }

                                val payload = packet.data.copyOfRange(headerEnd, headerEnd + payloadLen)
                                assembler.insert(seq, payload)

                                if (assembler.isComplete()) {
                                    val fullImage = assembler.build()
                                    imageBuffers.remove(fname)

                                    val bmp = BitmapFactory.decodeByteArray(fullImage, 0, fullImage.size) ?: continue

                                    val partsF = fname.split("_")
                                    val ts = partsF.getOrNull(0)?.toLongOrNull() ?: System.currentTimeMillis()
                                    val tid = partsF.getOrNull(1)?.toIntOrNull() ?: 0
                                    val cls = partsF.getOrNull(2) ?: "object"

                                    val gps = getNearestGps(System.currentTimeMillis())
                                    val detection = Detection(
                                        fname,
                                        cls,
                                        tid,
                                        ts,
                                        gps?.lat,
                                        gps?.lon,
                                        fullImage,
                                        bmp
                                    )

                                    synchronized(detections) {
                                        detections.add(0, detection)
                                        if (detections.size > 200) detections.removeLast()
                                    }

                                    saveImageToGallery(fname, bmp)

                                    if (getSupabaseEnabled()) {
                                        enqueueLogForUpload(fname, detection)
                                    }
                                }

                            } catch (e: SocketTimeoutException) {
                                // allows loop exit
                            } catch (e: Exception) {
                                Log.e("FW-IMG", "img recv error: ${e.message}")
                                Thread.sleep(10)
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("FW-IMG", "image socket fail: ${e.message}")
                }
            }
        }

        private fun saveImageToGallery(fname: String, bitmap: Bitmap) {
            try {
                val filename = "$fname.jpg"
                val resolver = this@MainActivity.contentResolver
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/MemoryRetrieve")
                    }
                }
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                uri?.let {
                    resolver.openOutputStream(it)?.use { out ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                    }
                }
            } catch (e: Exception) {
                Log.e("FW-GAL", "Exception saving $fname: ${e.message}")
            }
        }

        // ----------------- Simple in-memory upload queue for supabase (metadata+small logs)
        private val uploadQueue = Collections.synchronizedList(mutableListOf<JSONObject>())

        private fun enqueueLogForUpload(fname: String, detection: Detection) {
            try {
                val o = JSONObject()
                o.put("filename", fname)
                o.put("class", detection.className)
                o.put("ts", detection.ts)
                o.put("lat", detection.lat ?: JSONObject.NULL)
                o.put("lon", detection.lon ?: JSONObject.NULL)
                uploadQueue.add(o)
            } catch (e: Exception) {
                Log.e("FW-UPQ", "enqueue error: ${e.message}")
            }
        }

        private fun startSupabaseFlushLoop() {
            thread {
                while (running) {
                    try {
                        if (getSupabaseEnabled() && uploadQueue.isNotEmpty()) {
                            val key = prefs.getString(PREF_SUPABASE_KEY, null)
                            if (key.isNullOrBlank()) {
                                Log.w("FW-SUPA", "Supabase enabled but key missing in prefs. Set $PREF_SUPABASE_KEY")
                            } else {
                                flushUploadQueueToSupabase(key)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("FW-SUPA", "flush loop error: ${e.message}")
                    }
                    Thread.sleep(10_000) // every 10s if enabled
                }
            }
        }

        // Very small best-effort HTTP POST to Supabase (Project URL base must be set by user; we use path /rest/v1/logs as example).
        // WARNING: you must configure correct endpoint/headers on Supabase (this is a template).
        private fun flushUploadQueueToSupabase() {
            thread(name = "SupabaseUploader") {
                if (!getSupabaseEnabled()) return@thread
                if (uploadQueue.isEmpty()) return@thread

                val baseUrl = getSupabaseBaseUrl()
                val apiKey = getSupabaseApiKey()

                if (baseUrl.isNullOrBlank() || apiKey.isNullOrBlank()) {
                    Log.w("FW-SUPA", "Supabase not configured")
                    return@thread
                }

                val queueCopy = ArrayList(uploadQueue)

                for (entry in queueCopy) {
                    try {
                        val url = URL("$baseUrl/rest/v1/logs")
                        val conn = url.openConnection() as HttpURLConnection

                        conn.requestMethod = "POST"
                        conn.setRequestProperty("Content-Type", "application/json")
                        conn.setRequestProperty("apikey", apiKey)
                        conn.setRequestProperty("Authorization", "Bearer $apiKey")
                        conn.doOutput = true
                        conn.connectTimeout = 5000
                        conn.readTimeout = 5000

                        val json = JSONObject().apply {
                            put("filename", entry.filename)
                            put("class", entry.detection.cls)
                            put("track_id", entry.detection.trackId)
                            put("timestamp", entry.detection.timestamp)
                            put("lat", entry.detection.lat)
                            put("lon", entry.detection.lon)
                        }

                        conn.outputStream.use {
                            it.write(json.toString().toByteArray())
                        }

                        val code = conn.responseCode
                        if (code in 200..299) {
                            uploadQueue.remove(entry)   // ✅ remove only on success
                            Log.d("FW-SUPA", "Uploaded: ${entry.filename}")
                        } else {
                            Log.w("FW-SUPA", "Upload failed ($code), retry later")
                        }

                        conn.disconnect()

                    } catch (e: Exception) {
                        Log.w("FW-SUPA", "Upload error: ${e.message}")
                        break   // stop loop, retry later
                    }
                }
            }
        }

        private fun getSupabaseEnabled(): Boolean {
            return prefs.getBoolean(PREF_SUPABASE_ENABLED, false)
        }


        // ----------------- Command sender (to Pi) -----------------
        fun sendCommandToPi(cmd: String) {
            thread {
                try {
                    val target = piIp ?: return@thread
                    DatagramSocket().use { s ->
                        val msg = "$SECRET|$cmd".toByteArray()
                        s.send(DatagramPacket(msg, msg.size, InetAddress.getByName(target), COMMAND_PORT))
                    }
                } catch (e: Exception) {
                    Log.e("FW-CMD", "send error: ${e.message}")
                }
            }
        }

        // send to all known nodes (simple loop)
        fun sendCommandToAllNodes(cmd: String) {
            thread(name = "CMD-All-Nodes") {
                val nodes = nodeRegistry.values.toList()
                if (nodes.isEmpty()) return@thread

                try {
                    DatagramSocket().use { socket ->
                        val msg = "$SECRET|$cmd".toByteArray()
                        nodes.forEach { n ->
                            try {
                                val addr = InetAddress.getByName(n.ip)
                                socket.send(DatagramPacket(msg, msg.size, addr, COMMAND_PORT))
                            } catch (e: Exception) {
                                Log.w("FW-CMDALL", "Failed to send to ${n.ip}: ${e.message}")
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("FW-CMDALL", "Socket error: ${e.message}")
                }
            }
        }

        // ----------------- Pi AP watcher: scans for Pi Zero AP when offline and prompts phone UI -----------------
        private fun startPiApWatcher() {
            thread {
                try {
                    val wifiMgr = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                    while (running) {
                        try {
                            // Only actively scan if no internet or no known wifi
                            if (!hasInternet()) {
                                try { wifiMgr.startScan() } catch (_: Exception) {}
                                val results = try { wifiMgr.scanResults } catch (_: Exception) { emptyList<ScanResult>() }
                                val piAp = results.firstOrNull { r ->
                                    val s = r.SSID ?: ""
                                    s.startsWith("PI_ZERO") || s.startsWith("PI_AP") || s.startsWith("TRACKPI") || s.startsWith("PI5_RECOVERY")
                                }
                                if (piAp != null) {
                                    Log.d("FW-AP", "Found Pi AP: ${piAp.SSID}")
                                    // ask MainActivity to show popup
                                    this@MainActivity.runOnUiThread {
                                        openPortalActivity(html, lastPortalFromIp)
                                    }
                                    // Wait a bit so we don't spam UI
                                    Thread.sleep(8000)
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("FW-AP", "ap watcher error: ${e.message}")
                        }
                        Thread.sleep(3000)
                    }
                } catch (e: Exception) {
                    Log.e("FW-AP", "ap watcher setup failed: ${e.message}")
                }
            }
        }

        // ----------------- Simple network brain loop to enable offline scanning / recovery
        private fun networkBrainLoop() {
            thread {
                while (running) {
                    try {
                        if (!hasInternet()) {
                            // if there is no internet check for Pi Zero AP and the worker's Pi IP
                            // Pi Zero only saves/streams video (phone should only handle video)
                            // worker.startPiApWatcher() already triggers UI popup
                        } else {
                            // if internet present, ensure node registry is polled by heartbeat listener
                        }
                    } catch (e: Exception) {
                        Log.e("FW-NETB", "net brain error: ${e.message}")
                    }
                    Thread.sleep(5000)
                }
            }
        }
        // =================================================================================
// ====================== PHONE ORCHESTRATION CORE ================================
// =================================================================================

        private var phoneFallbackActive = false
        private var activeRoute: ActiveRoute? = null

        data class ActiveRoute(
            val streamNode: String,
            val computeNode: String,
            val outputNode: String
        )

        private fun orchestrationBrainLoop() {
            thread {
                while (running) {
                    try {
                        evaluateRouting()
                    } catch (e: Exception) {
                        Log.e("ORCH", "brain error: ${e.message}")
                    }
                    Thread.sleep(3000)
                }
            }
        }

        private fun evaluateRouting() {
            val nodes = nodeRegistry.values.toList()

            // If nothing alive → phone-only mode
            if (nodes.isEmpty()) {
                activatePhoneFallback()
                return
            }

            val compute = selectBestComputeNode(nodes)
            val stream = selectBestStreamNode(nodes)

            if (compute == null || stream == null) {
                activatePhoneFallback()
                return
            }

            val newRoute = ActiveRoute(
                streamNode = stream.id,
                computeNode = compute.id,
                outputNode = "PHONE"
            )

            if (activeRoute != newRoute) {
                applyRoute(newRoute)
            }
        }

        private fun applyRoute(route: ActiveRoute) {
            Log.i("ORCH", "Applying route: $route")

            activeRoute = route
            deactivatePhoneFallback()

            sendCommandToNode(route.streamNode, "STREAM_START")
            sendCommandToNode(route.computeNode, "COMPUTE_START")
        }
        private fun probeLatency(ip: String): Int {
            return try {
                val start = System.currentTimeMillis()
                val addr = InetAddress.getByName(ip)
                if (addr.isReachable(1000)) (System.currentTimeMillis() - start).toInt() else 999
            } catch (_: Exception) {
                999
            }
        }
        private fun selectBestComputeNode(nodes: List<NodeInfo>): NodeInfo? {
            return nodes
                .filter { it.role.uppercase() in listOf("LAPTOP", "PI5") }
                .maxByOrNull { computeNodeScore(it) }
        }

        private fun selectBestStreamNode(nodes: List<NodeInfo>): NodeInfo? {
            return nodes
                .filter { it.role.uppercase() in listOf("PI0", "PIZERO", "PI_ZERO") }
                .maxByOrNull { 100 - probeLatency(it.ip) }
        }

        private fun sendCommandToNode(nodeId: String, cmd: String) {
            val n = nodeRegistry[nodeId] ?: return
            sendUdp(n.ip, "$SECRET|$cmd")
        }

        private fun sendUdp(ip: String, msg: String) {
            thread {
                try {
                    val s = DatagramSocket()
                    try {
                        val data = msg.toByteArray()
                        s.send(DatagramPacket(data, data.size, InetAddress.getByName(ip), COMMAND_PORT))
                    } finally {
                        s.close()
                    }
                } catch (e: Exception) {
                    Log.e("ORCH", "send failed: ${e.message}")
                }
            }
        }

// =================================================================================
// ====================== PHONE CAMERA FALLBACK SYSTEM =============================
// =================================================================================

        private var cameraProvider: ProcessCameraProvider? = null
        private var imageAnalyzer: ImageAnalysis? = null
        private var cameraExecutor: ExecutorService? = null

        private fun activatePhoneFallback() {
            if (phoneFallbackActive) return

            phoneFallbackActive = true
            Log.w("PHONE", "Activating PHONE-ONLY MODE")
            startPhoneCamera()
        }

        private fun deactivatePhoneFallback() {
            if (!phoneFallbackActive) return

            phoneFallbackActive = false
            Log.w("PHONE", "Disabling PHONE-ONLY MODE")
            stopPhoneCamera()
        }

        private fun startPhoneCamera() {
            val future = ProcessCameraProvider.getInstance(this)
            future.addListener({
                cameraProvider = future.get()

                val preview = Preview.Builder().build()

                imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(getCameraExecutor()) { image ->
                            handlePhoneFrame(image)
                        }
                    }

                val selector = CameraSelector.DEFAULT_BACK_CAMERA

                try {
                    cameraProvider?.unbindAll()
                    cameraProvider?.bindToLifecycle(
                        this, selector, preview, imageAnalyzer
                    )
                } catch (e: Exception) {
                    Log.e("PHONE", "Camera bind failed: ${e.message}")
                }

            }, ContextCompat.getMainExecutor(this))
        }

        private fun stopPhoneCamera() {
            try {
                cameraProvider?.unbindAll()
            } catch (_: Exception) {}
        }

        private fun getCameraExecutor(): ExecutorService {
            if (cameraExecutor == null) {
                cameraExecutor = Executors.newSingleThreadExecutor()
            }
            return cameraExecutor!!
        }

        private fun handlePhoneFrame(image: ImageProxy) {
            try {
                // Placeholder for ML inference pipeline
                // Next step: add TensorFlow Lite model here

            } catch (e: Exception) {
                Log.e("PHONE", "Frame error: ${e.message}")
            } finally {
                image.close()
            }
        }

// =================================================================================
// ====================== FAILOVER + HEALTH MONITOR ================================
// =================================================================================

        private fun startFailoverMonitor() {
            thread {
                while (running) {
                    try {
                        checkNodeHealth()
                    } catch (e: Exception) {
                        Log.e("FAILOVER", "monitor error: ${e.message}")
                    }
                    Thread.sleep(5000)
                }
            }
        }

        private fun checkNodeHealth() {
            val now = System.currentTimeMillis()
            val deadNodes = nodeRegistry.values.filter {
                now - it.lastSeen > 6000
            }

            deadNodes.forEach {
                Log.w("FAILOVER", "Node lost: ${it.id}")
                nodeRegistry.remove(it.id)
            }

            if (deadNodes.isNotEmpty()) {
                Log.w("FAILOVER", "Node(s) lost: ${deadNodes.joinToString { it.id }}")
                // Immediately adjust routing using the existing brain
                evaluateRouting()
            }
        }
        private fun hasInternet(): Boolean {
            try {
                val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
                val net = cm.activeNetwork ?: return false
                val caps = cm.getNetworkCapabilities(net) ?: return false
                return caps.hasCapability(android.net.NetworkCapabilities.NET_CAPABILITY_INTERNET)
            } catch (e: Exception) {
                return false
            }
        }
        private fun computeNodeScore(n: NodeInfo): Int {
            var score = 0

            if (n.internet) score += 25
            if (n.role.uppercase() == "LAPTOP") score += 50
            if (n.role.uppercase() == "PI5") score += 35

            val cpuPenalty = n.load.coerceIn(0.0, 10.0)
            score += (10 - cpuPenalty).toInt()

            val latency = probeLatency(n.ip)
            score -= (latency / 40).toInt()

            return score
        }

        // ----------------- Portal handling: PortalActivity shows page and user completes auth,
        // then PortalActivity sends "PORTAL_AUTH_DONE" back to originating Pi by simple UDP.
        fun notifyPortalAuthDone(toIp: String?) {
            if (toIp == null) return
            thread {
                try {
                    DatagramSocket().use { s ->
                        val msg = "$SECRET|PORTAL_AUTH_DONE".toByteArray()
                        s.send(DatagramPacket(msg, msg.size, InetAddress.getByName(toIp), COMMAND_PORT))
                    }
                } catch (e: Exception) {
                    Log.e("FW-PORTAL", "notify fail: ${e.message}")
                }
            }
        }
    } // end ForegroundWorker class

    // ----------------- UI helpers for Pi connect popup & portal activity -----------------
    fun showPiConnectPopup(ssid: String) {
        try {
            val d = Dialog(this)
            d.requestWindowFeature(Window.FEATURE_NO_TITLE)
            d.setCancelable(true)
            d.window?.setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT)
            d.window?.setBackgroundDrawableResource(android.R.color.transparent)

            val outer = LinearLayout(this)
            outer.orientation = LinearLayout.VERTICAL
            outer.setBackgroundColor(Color.parseColor("#80000000")) // semi-transparent
            outer.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT)

            val card = LinearLayout(this)
            card.orientation = LinearLayout.HORIZONTAL
            card.setBackgroundColor(Color.WHITE)
            card.elevation = 12f
            val pad = (16 * resources.displayMetrics.density).toInt()
            card.setPadding(pad, pad, pad, pad)

            val img = ImageView(this)
            val drawableId = try { resources.getIdentifier("pi_zero", "drawable", packageName) } catch (_: Exception) { 0 }
            if (drawableId != 0) img.setImageResource(drawableId) else img.setImageResource(android.R.drawable.sym_def_app_icon)
            val ivparams = LinearLayout.LayoutParams((64 * resources.displayMetrics.density).toInt(), (64 * resources.displayMetrics.density).toInt())
            img.layoutParams = ivparams
            img.scaleType = ImageView.ScaleType.FIT_CENTER

            val meta = LinearLayout(this)
            meta.orientation = LinearLayout.VERTICAL
            meta.setPadding((12 * resources.displayMetrics.density).toInt(), 0, 0, 0)
            meta.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

            val title = TextView(this)
            title.text = ssid
            title.setTextColor(Color.BLACK)
            title.textSize = 16f

            val subtitle = TextView(this)
            subtitle.text = "Raspberry Pi detected • Tap to connect"
            subtitle.setTextColor(Color.DKGRAY)
            subtitle.textSize = 12f

            val btn = Button(this)
            btn.text = "Connect"
            btn.setOnClickListener {
                // Try programmatic connect on Q+ otherwise open Wi-Fi settings
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    connectToPiApUsingSpecifier(ssid)
                } else {
                    startActivity(Intent(android.provider.Settings.ACTION_WIFI_SETTINGS))
                }
                d.dismiss()
            }

            meta.addView(title)
            meta.addView(subtitle)
            meta.addView(btn)

            card.addView(img)
            card.addView(meta)

            val wrapper = LinearLayout(this)
            wrapper.orientation = LinearLayout.VERTICAL
            wrapper.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT)
            wrapper.gravity = Gravity.BOTTOM
            val cardLp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            cardLp.setMargins(20, 0, 20, 40)
            wrapper.addView(card, cardLp)

            outer.addView(wrapper)
            d.setContentView(outer)

            d.show()
        } catch (e: Exception) {
            Log.e("MA-POP", "popup failed: ${e.message}")
        }
    }

    // Programmatic connect using WifiNetworkSpecifier (Android Q+). Binds process to network for portal auth.
    private var boundNetworkCallback: ConnectivityManager.NetworkCallback? = null
    @SuppressLint("MissingPermission")
    private fun connectToPiApUsingSpecifier(ssid: String) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            startActivity(Intent(android.provider.Settings.ACTION_WIFI_SETTINGS))
            return
        }
        try {
            val spec = WifiNetworkSpecifier.Builder()
                .setSsid(ssid)
                .build()
            val req = NetworkRequest.Builder()
                .addTransportType(android.net.NetworkCapabilities.TRANSPORT_WIFI)
                .setNetworkSpecifier(spec)
                .build()
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            // unregister previous
            try { boundNetworkCallback?.let { cm.unregisterNetworkCallback(it) } } catch (_: Exception) {}
            boundNetworkCallback = object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    try {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            cm.bindProcessToNetwork(network)
                        } else {
                            @Suppress("DEPRECATION")
                            ConnectivityManager.setProcessDefaultNetwork(network)
                        }
                        showToast("Bound to $ssid for portal auth")
                        // notify Pi that phone bound (simple UDP ping)
                        thread {
                            try {
                                DatagramSocket().use { s ->
                                    val msg = "$SECRET|PHONE_BOUND".toByteArray()
                                    s.send(DatagramPacket(msg, msg.size, InetAddress.getByName("255.255.255.255"), COMMAND_PORT))
                                }
                            } catch (_: Exception) {}
                        }
                    } catch (e: Exception) {
                        Log.w("MA-WIFI", "bind failed: ${e.message}")
                    }
                }
                override fun onUnavailable() {
                    showToast("Failed to connect to $ssid")
                }
                override fun onLost(network: Network) {
                    showToast("$ssid disconnected")
                    try {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            cm.bindProcessToNetwork(null)
                        } else {
                            @Suppress("DEPRECATION")
                            ConnectivityManager.setProcessDefaultNetwork(null)
                        }
                    } catch (_: Exception) {}
                }
            }
            cm.requestNetwork(req, boundNetworkCallback!!)
        } catch (e: Exception) {
            Log.e("MA-WIFI", "connect spec err: ${e.message}")
            startActivity(Intent(android.provider.Settings.ACTION_WIFI_SETTINGS))
        }
    }

    // PortalActivity opener
    fun openPortalActivity(html: String, fromIp: String?) {
        try {
            val i = Intent(this, PortalActivity::class.java)
            i.flags = Intent.FLAG_ACTIVITY_NEW_TASK
            i.putExtra("portal_html", html)
            i.putExtra("from_ip", fromIp)
            startActivity(i)
        } catch (e: Exception) {
            Log.e("MA-PORT", "open portal act: ${e.message}")
        }
    }

    // ----------------- PortalActivity to display portal HTML and report back completion -----------------
    class PortalActivity : Activity() {
        private var portalHtml: String? = null
        private var fromIp: String? = null
        private val SECRET = "memoryretrieve##$"
        private val COMMAND_PORT = 5005

        override fun onCreate(savedInstanceState: Bundle?) {

            super.onCreate(savedInstanceState)
            try {
                portalHtml = intent.getStringExtra("portal_html")
                fromIp = intent.getStringExtra("from_ip")
            } catch (_: Exception) {}

            val layout = LinearLayout(this)
            layout.orientation = LinearLayout.VERTICAL
            val web = WebView(this)
            web.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f)
            web.settings.javaScriptEnabled = true
            web.webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) { super.onPageFinished(view, url) }
            }

            val btn = Button(this)
            btn.text = "Done — Notify Pi"
            btn.setOnClickListener {
                thread {
                    try {
                        val target = fromIp ?: return@thread
                        DatagramSocket().use { s ->
                            val msg = "$SECRET|PORTAL_AUTH_DONE".toByteArray(Charsets.UTF_8)
                            s.send(DatagramPacket(msg, msg.size, InetAddress.getByName(target), COMMAND_PORT))
                        }
                    } catch (e: Exception) {
                        Log.e("PORTAL-ACT", "notify pi failed: ${e.message}")
                    }
                }
                finish()
            }

            layout.addView(web)
            layout.addView(btn)
            setContentView(layout)

            if (!portalHtml.isNullOrBlank()) {
                try {
                    web.loadDataWithBaseURL(null, portalHtml!!, "text/html", "utf-8", null)
                } catch (e: Exception) {
                    Log.e("PORTAL-ACT", "load html failed: ${e.message}")
                }
            } else {
                web.loadUrl("about:blank")
            }
        }
    }

    // ----------------- Utility HMAC (if needed later) -----------------
    private fun hmacSha256Hex(key: ByteArray, msg: ByteArray): String {
        return try {
            val mac = Mac.getInstance("HmacSHA256")
            val sk = SecretKeySpec(key, "HmacSHA256")
            mac.init(sk)
            val digest = mac.doFinal(msg)
            digest.joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            Log.e("MA-HMAC", "err: ${e.message}")
            ""
        }
    }
    private fun requestRuntimePermissions() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_WIFI_STATE,
                Manifest.permission.CHANGE_WIFI_STATE
            ),
            1
        )
    }
}
