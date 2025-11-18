package ru.sr.core.crypto

import kotlinx.coroutines.flow.Flow

interface IKeyStore {
    suspend fun getOrCreateIdentity(): IdentityKeyPair
    suspend fun saveDevice(device: DeviceKey)
    suspend fun revokeDevice(deviceId: String)
    fun devices(): Flow<List<DeviceKey>>
}

data class IdentityKeyPair(
    val identityKey: ByteArray,
    val signedPreKey: ByteArray,
    val signature: ByteArray
)

data class DeviceKey(
    val id: String,
    val publicKey: ByteArray,
    val label: String,
    val addedAt: Long
)

interface ISession {
    val peerId: String
    suspend fun encrypt(message: ByteArray): CipherMessage
    suspend fun decrypt(cipher: CipherMessage): ByteArray
    suspend fun ratchet()
}

data class CipherMessage(
    val header: ByteArray,
    val body: ByteArray,
    val mac: ByteArray
)

interface IGroupSession {
    val groupId: String
    suspend fun encrypt(message: ByteArray): CipherMessage
    suspend fun decrypt(cipher: CipherMessage): ByteArray
}

interface ICryptoService {
    suspend fun createSession(peerId: String): ISession
    suspend fun getSession(peerId: String): ISession?
    suspend fun closeSession(peerId: String)
    suspend fun createGroupSession(groupId: String): IGroupSession
    fun fingerprintFor(peerId: String): String
}
