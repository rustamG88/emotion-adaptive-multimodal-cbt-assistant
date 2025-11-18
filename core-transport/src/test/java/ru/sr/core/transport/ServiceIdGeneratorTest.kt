package ru.sr.core.transport

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.security.MessageDigest

class ServiceIdGeneratorTest {
    @Test
    fun `generator returns truncated sha256 digest`() {
        val zone = "Офис-5э"
        val topic = "Связь"
        val expected = MessageDigest.getInstance("SHA-256")
            .digest("sr|ОФИС-5Э|СВЯЗЬ".toByteArray())
            .joinToString(separator = "") { byte -> "%02x".format(byte) }
            .take(20)

        val serviceId = ServiceIdGenerator.generate(zone, topic)

        assertEquals(20, serviceId.length)
        assertEquals(expected, serviceId)
    }

    @Test
    fun `generator falls back to default when blank`() {
        val serviceId = ServiceIdGenerator.generate("   ", " ")
        assertEquals(
            ServiceIdGenerator.generate("DEFAULT", "DEFAULT"),
            serviceId
        )
        assertTrue(serviceId.isNotBlank())
    }
}
