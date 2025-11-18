package ru.sr.core.transport

import java.security.MessageDigest
import java.util.Locale

/**
 * Generates deterministic Nearby service identifiers based on zone and topic tags.
 * The output is a hex-encoded SHA-256 digest truncated to 20 characters as required by Nearby.
 */
object ServiceIdGenerator {
    private const val DEFAULT_VALUE = "DEFAULT"

    fun generate(zone: String, topic: String): String {
        val normalizedZone = zone.normalize()
        val normalizedTopic = topic.normalize()
        val payload = "sr|$normalizedZone|$normalizedTopic"
        val digest = MessageDigest.getInstance("SHA-256").digest(payload.toByteArray())
        return digest.toHex().take(20)
    }

    private fun String.normalize(): String = trim()
        .ifBlank { DEFAULT_VALUE }
        .uppercase(Locale.ROOT)

    private fun ByteArray.toHex(): String = joinToString(separator = "") { byte ->
        "%02x".format(byte)
    }
}
