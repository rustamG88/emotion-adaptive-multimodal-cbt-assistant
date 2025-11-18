package ru.sr.core.crypto

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.security.MessageDigest
import java.util.UUID

class FakeCryptoService : ICryptoService, IKeyStore {
    private val sessions = mutableMapOf<String, FakeSession>()
    private val devicesFlow = MutableStateFlow<List<DeviceKey>>(emptyList())
    private val lock = Mutex()

    override suspend fun createSession(peerId: String): ISession = lock.withLock {
        sessions.getOrPut(peerId) { FakeSession(peerId) }
    }

    override suspend fun getSession(peerId: String): ISession? = lock.withLock {
        sessions[peerId]
    }

    override suspend fun closeSession(peerId: String) {
        lock.withLock {
            sessions.remove(peerId)
        }
    }

    override suspend fun createGroupSession(groupId: String): IGroupSession = FakeGroupSession(groupId)

    override fun fingerprintFor(peerId: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(peerId.toByteArray())
        val emojis = listOf("🛰️", "🔐", "📡", "🤝", "🛡️", "💬", "🌐", "⚡", "🧭", "📶", "🪐", "✨", "🔁", "🪄", "🛰", "🧿")
        val builder = StringBuilder()
        digest.take(4).forEach { byte ->
            val idx = (byte.toInt() and 0xFF) % emojis.size
            builder.append(emojis[idx])
        }
        return builder.toString()
    }

    override suspend fun getOrCreateIdentity(): IdentityKeyPair {
        val seed = UUID.randomUUID().toString().encodeToByteArray()
        return IdentityKeyPair(seed, seed.reversedArray(), seed.copyOfRange(0, 16))
    }

    override suspend fun saveDevice(device: DeviceKey) {
        lock.withLock {
            val updated = devicesFlow.value.filterNot { it.id == device.id } + device
            devicesFlow.value = updated
        }
    }

    override suspend fun revokeDevice(deviceId: String) {
        lock.withLock {
            devicesFlow.value = devicesFlow.value.filterNot { it.id == deviceId }
        }
    }

    override fun devices(): Flow<List<DeviceKey>> = devicesFlow.asStateFlow()

    private class FakeSession(override val peerId: String) : ISession {
        private val nonceCounter = MutableStateFlow(0)

        override suspend fun encrypt(message: ByteArray): CipherMessage {
            val nonce = nonceCounter.value + 1
            nonceCounter.value = nonce
            return CipherMessage(
                header = "nonce:$nonce".encodeToByteArray(),
                body = message.reversedArray(),
                mac = peerId.encodeToByteArray()
            )
        }

        override suspend fun decrypt(cipher: CipherMessage): ByteArray = cipher.body.reversedArray()

        override suspend fun ratchet() {
            nonceCounter.value += 1
        }
    }

    private class FakeGroupSession(override val groupId: String) : IGroupSession {
        override suspend fun encrypt(message: ByteArray): CipherMessage = CipherMessage(
            header = groupId.encodeToByteArray(),
            body = message,
            mac = groupId.encodeToByteArray()
        )

        override suspend fun decrypt(cipher: CipherMessage): ByteArray = cipher.body
    }
}
