package ru.sr.core.storage.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey val id: String,
    val conversationId: String,
    val senderId: String,
    val cipherText: ByteArray,
    val timestamp: Long,
    val delivered: Boolean,
    val ttl: Int
)

@Entity(tableName = "peers")
data class PeerEntity(
    @PrimaryKey val peerId: String,
    val nickname: String?,
    val hops: Int,
    val proximity: String,
    val lastSeen: Long,
    val supportsDirect: Boolean
)

@Entity(tableName = "sessions")
data class SessionEntity(
    @PrimaryKey val peerId: String,
    val rootKey: ByteArray,
    val createdAt: Long,
    val updatedAt: Long
)

@Entity(tableName = "devices")
data class DeviceEntity(
    @PrimaryKey val id: String,
    val label: String,
    val addedAt: Long,
    val signature: ByteArray
)
