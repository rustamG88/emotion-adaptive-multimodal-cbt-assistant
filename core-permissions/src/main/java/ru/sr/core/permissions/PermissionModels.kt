package ru.sr.core.permissions

enum class PermissionType {
    Bluetooth,
    NearbyWifi,
    ForegroundService,
    Notifications
}

data class PermissionStatus(
    val type: PermissionType,
    val granted: Boolean,
    val shouldShowRationale: Boolean
)
