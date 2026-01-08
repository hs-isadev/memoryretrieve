package com.trackit.myapplication

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
        val imageBytes: ByteArray
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
                val bmp = BitmapFactory.decodeByteArray(d.imageBytes, 0, d.imageBytes.size)
                view.findViewById<ImageView>(R.id.imgThumb).setImageBitmap(bmp)
                view.findViewById<TextView>(R.id.textClass).text = d.className.uppercase()
                view.findViewById<TextView>(R.id.textTrack).text = "ID: ${d.trackId}"
                view.setOnClickListener {
                    d.lat?.let { lat -> d.lon?.let { lon ->
                        startActivity(android.content.Intent(android.content.Intent.ACTION_VIEW,
                            Uri.parse("geo:0,0?q=$lat,$lon(${d.className})")))
                    }}
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


    private fun startUdpListeners() {
        // Live video feed
        thread {
            DatagramSocket(VIDEO_PORT).use { s ->
                val buf = ByteArray(65507)
                while (true) {
                    try {
                        val p = DatagramPacket(buf, buf.size)
                        s.receive(p)
                        val bmp = BitmapFactory.decodeByteArray(p.data, 0, p.length)
                        handler.post { imgLive.setImageBitmap(bmp) }
                    } catch (_: Exception) {}
                }
            }
        }

        // Image chunks (Option C format)
        thread {
            DatagramSocket(IMAGE_PORT).use { s ->
                val buf = ByteArray(65507)
                while (true) {
                    try {
                        val p = DatagramPacket(buf, buf.size)
                        s.receive(p)
                        val text = String(p.data, 0, p.length)
                        if (text.startsWith("IMG|")) {
                            val parts = text.split("|")
                            if (parts.size >= 6) {
                                val fname = parts[1]
                                val seq = parts[2].toInt()
                                val total = parts[3].toInt()
                                val crc = parts[4].toInt()
                                val len = parts[5].toInt()

                                val header = "IMG|$fname|$seq|$total|$crc|$len|".toByteArray()
                                val payload = p.data.copyOfRange(header.size, header.size + len)

                                val buffer = imageBuffers.getOrPut(fname) { ImageBuffer(total) }
                                buffer.chunks[seq] = payload

                                if (buffer.chunks.size == total) {
                                    val fullImage = buffer.chunks.toSortedMap().values.reduce { a, b -> a + b }
                                    imageBuffers.remove(fname)
                                    processFullImage(fname, fullImage)
                                }
                            }
                        }
                    } catch (e: Exception) { e.printStackTrace() }
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

    private fun startGpsLogging() {
        thread {
            while (true) {
                try {
                    @SuppressLint("MissingPermission")
                    val loc = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                        ?: locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                    if (loc != null) {
                        gpsLog.add(GPSEntry(System.currentTimeMillis(), loc.latitude, loc.longitude))
                        if (gpsLog.size > 1000) gpsLog.removeAt(0)
                    }
                } catch (e: Exception) { e.printStackTrace() }
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
