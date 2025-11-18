package ru.sr.core.testing

import kotlinx.coroutines.Dispatchers
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TestDispatcherProviderTest {
    @Test
    fun `default provider exposes dispatchers`() {
        val provider = DefaultDispatcherProvider()
        assertEquals(Dispatchers.IO, provider.io)
        assertEquals(Dispatchers.Default, provider.default)
    }
}
