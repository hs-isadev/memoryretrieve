package com.trackit.myapplication
import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.content.Intent
import android.Manifest
import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import java.net.*
import java.util.Collections
import kotlin.collections.HashMap
import kotlin.concurrent.thread
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

class MainActivity : AppCompatActivity() {

    private lateinit var imgLive: ImageView
    private lateinit var grid: GridView
    private lateinit var etSearch: EditText
    private lateinit var btnSearch: Button
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var tvStatus: TextView

    private var piIp: String? = null
    private val SECRET = "memoryretrieve##$"
    private val COMMAND_PORT = 5005
    private val VIDEO_PORT = 6000
    private val IMAGE_PORT = 6200
    private val DISCOVERY_PORT = 5004

    private val handler = Handler(Looper.getMainLooper())

    private val gpsLog = Collections.synchronizedList(mutableListOf<GPSEntry>())
    private lateinit var locationManager: LocationManager

    data class GPSEntry(val ts: Long, val lat: Double, val lon: Double)

    data class Detection(
        val filename: String,
        val className: String,
        val trackId: Int,
        val ts: Long,
        val lat: Double?,
        val lon: Double?,
        val imageBytes: ByteArray,
        var bitmap: Bitmap? = null,
        val boxes: MutableList<BBox> = mutableListOf()
    )

    data class BBox(
        val x1: Int,
        val y1: Int,
        val x2: Int,
        val y2: Int,
        val label: String,
        val trackId: Int
    )



    private val detections = Collections.synchronizedList(mutableListOf<Detection>())
    private val filtered = mutableListOf<Detection>()

    private val imageBuffers = HashMap<String, ImageBuffer>()
    data class ImageBuffer(var totalChunks: Int = 0, val chunks: HashMap<Int, ByteArray> = HashMap())

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
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 1)

        startGpsLogging()
        startDiscovery()
        startUdpListeners()
        startImageListener()
        startMetadataListener()


        btnStart.setOnClickListener { sendCommand("START") }
        btnStop.setOnClickListener { sendCommand("STOP") }
        btnSearch.setOnClickListener { filterDetections() }

        grid.adapter = object : BaseAdapter() {
            override fun getCount() = filtered.size
            override fun getItem(p: Int) = filtered[p]
            override fun getItemId(p: Int) = p.toLong()
            override fun getView(p: Int, v: android.view.View?, parent: android.view.ViewGroup?): android.view.View {
                val view = v ?: layoutInflater.inflate(R.layout.item_detection, parent, false)
                val d = filtered[p]

                // Use stored bitmap
                val baseBmp = d.bitmap ?: BitmapFactory.decodeByteArray(
                    d.imageBytes, 0, d.imageBytes.size
                )

                val mutable = baseBmp.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val paint = Paint().apply {
                    color = Color.GREEN
                    style = Paint.Style.STROKE
                    strokeWidth = 4f
                    textSize = 28f
                }

                for (b in d.boxes) {
                    canvas.drawRect(
                        b.x1.toFloat(),
                        b.y1.toFloat(),
                        b.x2.toFloat(),
                        b.y2.toFloat(),
                        paint
                    )
                    canvas.drawText(
                        "${b.label}_${b.trackId}",
                        b.x1.toFloat() + 4,
                        b.y1.toFloat() - 6,
                        paint
                    )
                }

                view.findViewById<ImageView>(R.id.imgThumb).setImageBitmap(mutable)


                view.findViewById<TextView>(R.id.textClass).text = d.className.uppercase()
                view.findViewById<TextView>(R.id.textTrack).text = "ID: ${d.trackId}"

                view.setOnClickListener {
                    d.lat?.let { lat ->
                        d.lon?.let { lon ->
                            startActivity(
                                Intent(
                                    Intent.ACTION_VIEW,
                                    Uri.parse("geo:0,0?q=$lat,$lon(${d.className})")
                                )
                            )
                        }
                    }
                }

                return view

            }
        }

    }

    private fun sendCommand(cmd: String) {
        piIp?.let { ip ->
            thread {
                try {
                    DatagramSocket().use { s ->
                        val msg = "$SECRET|$cmd".toByteArray()
                        s.send(DatagramPacket(msg, msg.size, InetAddress.getByName(ip), COMMAND_PORT))
                    }
                } catch (e: Exception) { e.printStackTrace() }
            }
        }
    }
    private fun startMetadataListener() {
        thread {
            DatagramSocket(META_PORT).use { socket ->
                val buf = ByteArray(8192)

                while (true) {
                    try {
                        val packet = DatagramPacket(buf, buf.size)
                        socket.receive(packet)

                        val msg = String(packet.data, 0, packet.length)
                        val json = org.json.JSONObject(msg)

                        if (json.optString("type") != "bbox") continue

                        val fname = json.getString("filename")
                        val bbox = json.getJSONArray("bbox")

                        val box = BBox(
                            x1 = bbox.getInt(0),
                            y1 = bbox.getInt(1),
                            x2 = bbox.getInt(2),
                            y2 = bbox.getInt(3),
                            label = json.getString("classname"),
                            trackId = json.getInt("trackId")
                        )

                        handler.post {
                            detections.find { it.filename == fname }
                                ?.boxes
                                ?.add(box)
                        }

                    } catch (e: Exception) {
                        Log.e("META", "bbox recv error", e)
                    }
                }
            }
        }
    }

    private fun startDiscovery() {
        thread {
            try {
                // Use a single socket bound to DISCOVERY_PORT
                DatagramSocket(DISCOVERY_PORT).use { socket ->
                    socket.broadcast = true
                    val buf = ByteArray(1024)
                    val secretMsg = "$SECRET|DISCOVER".toByteArray()

                    // Thread to receive Pi replies
                    thread {
                        while (true) {
                            try {
                                val packet = DatagramPacket(buf, buf.size)
                                socket.receive(packet)
                                val msg = String(packet.data, 0, packet.length)
                                if (msg.startsWith("OK|PI|")) {
                                    piIp = msg.substring(6)
                                    handler.post { tvStatus.text = "Connected to Pi: $piIp" }
                                }
                            } catch (_: Exception) { }
                        }
                    }

                    // Broadcast discovery periodically
                    while (true) {
                        try {
                            val broadcastAddr = InetAddress.getByName("255.255.255.255")
                            val packet = DatagramPacket(secretMsg, secretMsg.size, broadcastAddr, DISCOVERY_PORT)
                            socket.send(packet)
                        } catch (_: Exception) { }
                        Thread.sleep(3000)
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private val LOG_PORT = 9100
    private val META_PORT = 6201

    private fun startUdpListeners() {
        // --- LIVE VIDEO FEED ---
        thread {
            DatagramSocket(VIDEO_PORT).use { s ->
                val buf = ByteArray(65507)
                while (true) {
                    try {
                        val p = DatagramPacket(buf, buf.size)
                        s.receive(p)
                        val bmp = BitmapFactory.decodeByteArray(p.data, 0, p.length)
                        handler.post { imgLive.setImageBitmap(bmp) }
                    } catch (e: Exception) {
                        Log.e("UDP_VIDEO", "Error receiving video: ${e.message}")
                    }
                }
            }
        }

        // --- LOG LISTENER ---
        thread {
            DatagramSocket(LOG_PORT).use { s ->
                val buf = ByteArray(16384) // increase buffer size
                while (true) {
                    try {
                        val p = DatagramPacket(buf, buf.size)
                        s.receive(p)
                        val msg = String(p.data, 0, p.length, Charsets.UTF_8)
                        Log.d("PiLOG", "Received: $msg")
                    } catch (e: Exception) {
                        Log.e("PiLOG", "Listener error: ${e.message}")
                    }
                }
            }
        }


    }



    private fun processFullImage(filename: String, bytes: ByteArray) {
        val ts = System.currentTimeMillis()
        val gps = getNearestGps(ts)
        val parts = filename.split("_")
        val trackId = parts.getOrNull(1)?.toIntOrNull() ?: 0
        val className = parts.getOrNull(2)?.removeSuffix(".jpg") ?: "object"

        handler.post {
            detections.add(0, Detection(filename, className, trackId, ts, gps?.lat, gps?.lon, bytes))
            filterDetections()
        }
    }
    private fun processFullImageWithBoxes(
        filename: String, bitmap: Bitmap, bboxList: List<Map<String, Any>> = emptyList()
    ) {
        try {
            val bmpCopy = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(bmpCopy)
            val paint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 3f
                textSize = 24f
            }
            val textBg = Paint().apply {
                color = Color.argb(150, 0, 0, 0)
                style = Paint.Style.FILL
            }

            for (d in bboxList) {
                val x1 = (d["x1"] as? Number)?.toFloat() ?: 0f
                val y1 = (d["y1"] as? Number)?.toFloat() ?: 0f
                val x2 = (d["x2"] as? Number)?.toFloat() ?: 0f
                val y2 = (d["y2"] as? Number)?.toFloat() ?: 0f
                val className = d["classname"] as? String ?: "obj"
                val trackId = (d["trackId"] as? Number)?.toInt() ?: 0

                val rect = android.graphics.RectF(x1, y1, x2, y2)
                canvas.drawRect(rect, paint)
                canvas.drawRect(rect.left, rect.top - 28f, rect.left + paint.measureText("${className}_${trackId}") + 8f, rect.top, textBg)
                canvas.drawText("${className}_${trackId}", rect.left + 4f, rect.top - 4f, paint)
            }

            saveImageToGallery(filename, bmpCopy)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }


    private fun startImageListener() {
        thread {
            DatagramSocket(IMAGE_PORT).use { socket ->
                val buf = ByteArray(65507)
                val imageBuffers = HashMap<String, ImageBuffer>()

                while (true) {
                    try {
                        val packet = DatagramPacket(buf, buf.size)
                        socket.receive(packet)

                        // ---- find first 6 '|' to delimit header ----
                        val pipeIdxs = ArrayList<Int>(6)
                        for (i in 0 until packet.length) {
                            if (packet.data[i].toInt() == '|'.code) {
                                pipeIdxs.add(i)
                                if (pipeIdxs.size == 6) break
                            }
                        }
                        if (pipeIdxs.size < 6) continue

                        // ---- parse header ----
                        val headerEnd = pipeIdxs[5] + 1
                        val headerStr = String(packet.data, 0, headerEnd, Charsets.UTF_8)
                        val parts = headerStr.split("|")
                        if (parts.size < 6 || parts[0] != "IMG") continue

                        val fname = parts[1]
                        val seq = parts[2].toInt()
                        val total = parts[3].toInt()
                        val payloadLen = parts[5].toInt()

                        // ---- extract payload safely ----
                        if (headerEnd + payloadLen > packet.length) continue
                        val payload = packet.data.copyOfRange(
                            headerEnd,
                            headerEnd + payloadLen
                        )

                        // ---- store chunk ----
                        val buffer = imageBuffers.getOrPut(fname) {
                            ImageBuffer(total)
                        }
                        buffer.chunks[seq] = payload

                        // ---- reconstruct when complete ----
                        if (buffer.chunks.size == total) {
                            val fullImage = buffer.chunks
                                .toSortedMap()
                                .values
                                .reduce { a, b -> a + b }

                            imageBuffers.remove(fname)

                            val bmp = BitmapFactory.decodeByteArray(
                                fullImage, 0, fullImage.size
                            ) ?: continue

                            handler.post {
                                detections.add(
                                    0,
                                    Detection(
                                        filename = fname,
                                        className = fname.split("_").getOrNull(2) ?: "object",
                                        trackId = fname.split("_").getOrNull(1)?.toIntOrNull() ?: 0,
                                        ts = fname.split("_").getOrNull(0)?.toLongOrNull()
                                            ?: System.currentTimeMillis(),
                                        lat = getNearestGps(System.currentTimeMillis())?.lat,
                                        lon = getNearestGps(System.currentTimeMillis())?.lon,
                                        imageBytes = fullImage,
                                        bitmap = bmp
                                    )
                                )
                                filterDetections()
                                grid.smoothScrollToPosition(0)
                                saveImageToGallery(fname, bmp)
                            }
                        }

                    } catch (e: Exception) {
                        Log.e("UDP_IMAGE", "Listener error", e)
                    }
                }
            }
        }
    }



    private fun saveImageToGallery(fname: String, bitmap: Bitmap) {
        try {
            val filename = "$fname.jpg"
            val resolver = contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/MemoryRetrieve")
                }
            }

            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            if (uri != null) {
                resolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                // Force media scanner to detect new file
                sendBroadcast(Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, uri))
                Log.d("GALLERY", "Saved $filename to gallery")
            } else {
                Log.e("GALLERY", "Failed to insert $filename")
            }

        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("GALLERY", "Exception saving $fname: ${e.message}")
        }
    }


    private fun startGpsLogging() {
        thread {
            while (true) {
                try {
                    @SuppressLint("MissingPermission")
                    val loc = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                        ?: locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                    if (loc != null) {
                        gpsLog.add(GPSEntry(System.currentTimeMillis(), loc.latitude, loc.longitude))
                        if (gpsLog.size > 1000) gpsLog.removeAt(0) // keep latest 1000 points
                    }
                } catch (e: Exception) {
                    Log.e("GPS_LOG", "Error reading location: ${e.message}")
                }
                Thread.sleep(1500)
            }
        }
    }


    private fun getNearestGps(ts: Long): GPSEntry? {
        return gpsLog.minByOrNull { kotlin.math.abs(it.ts - ts) }
    }

    private fun filterDetections() {
        val query = etSearch.text.toString().lowercase()
        filtered.clear()
        filtered.addAll(if (query.isBlank()) detections else detections.filter { it.className.lowercase().contains(query) })
        (grid.adapter as BaseAdapter).notifyDataSetChanged()
    }
}
