package ru.sr.core.mesh

import kotlinx.serialization.Serializable
import java.util.UUID

@Serializable
data class MeshFrame(
    val header: MeshHeader,
    val body: ByteArray,
    val signature: ByteArray
)

@Serializable
data class MeshHeader(
    val id: String = UUID.randomUUID().toString(),
    val ttl: Int = 6,
    val hop: Int = 0,
    val frameClass: FrameClass = FrameClass.TEXT,
    val topicTag: String? = null
)

enum class FrameClass(val code: Int) {
    TEXT(1),
    ACK(2),
    CTRL(3),
    MEDIA_META(10),
    MEDIA_CHUNK(11)
}

fun MeshHeader.nextHop(): MeshHeader = copy(ttl = ttl - 1, hop = hop + 1)
