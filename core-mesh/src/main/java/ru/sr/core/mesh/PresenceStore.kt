package ru.sr.core.mesh

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.absoluteValue

class PresenceStore {
    private val lock = Mutex()
    private val peersFlow = MutableStateFlow<List<PeerPresence>>(emptyList())

    val peers: Flow<List<PeerPresence>> = peersFlow.asStateFlow()

    suspend fun update(presence: PeerPresence) {
        lock.withLock {
            val current = peersFlow.value.toMutableList()
            val index = current.indexOfFirst { it.peerId == presence.peerId }
            if (index >= 0) {
                current[index] = presence
            } else {
                current += presence
            }
            peersFlow.value = current.sortedBy { it.nickname ?: "Пользователь" }
        }
    }

    suspend fun prune(staleThresholdMillis: Long, now: Long) {
        lock.withLock {
            peersFlow.value = peersFlow.value.filter { now - it.lastSeen <= staleThresholdMillis }
        }
    }

    fun proximityFromRssi(rssi: Int): Proximity = when {
        rssi >= -60 -> Proximity.NEAR
        rssi in -75..-61 -> Proximity.WALL
        else -> Proximity.FAR
    }
}

data class PeerPresence(
    val peerId: String,
    val nickname: String?,
    val hops: Int,
    val proximity: Proximity,
    val lastSeen: Long,
    val supportsDirect: Boolean,
    val online: Boolean
)

enum class Proximity { NEAR, WALL, FAR }
