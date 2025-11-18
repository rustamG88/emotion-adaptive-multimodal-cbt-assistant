package ru.sr.core.permissions

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

class PermissionCoordinator(private val context: Context) {
    fun currentStatus(type: PermissionType): PermissionStatus {
        val permission = when (type) {
            PermissionType.Bluetooth -> Manifest.permission.BLUETOOTH_SCAN
            PermissionType.NearbyWifi -> Manifest.permission.NEARBY_WIFI_DEVICES
            PermissionType.ForegroundService -> Manifest.permission.FOREGROUND_SERVICE_DATA_SYNC
            PermissionType.Notifications -> Manifest.permission.POST_NOTIFICATIONS
        }
        val granted = ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        val rationale = if (context is Activity) {
            context.shouldShowRequestPermissionRationale(permission)
        } else {
            false
        }
        return PermissionStatus(type, granted, rationale)
    }

    fun requestLauncher(activity: Activity, callback: (Map<String, Boolean>) -> Unit): ActivityResultLauncher<Array<String>> {
        return activity.registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions(), callback)
    }

    fun permissionsFor(type: PermissionType): Array<String> = when (type) {
        PermissionType.Bluetooth -> arrayOf(
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_ADVERTISE,
            Manifest.permission.BLUETOOTH_CONNECT
        )
        PermissionType.NearbyWifi -> arrayOf(Manifest.permission.NEARBY_WIFI_DEVICES)
        PermissionType.ForegroundService -> arrayOf(Manifest.permission.FOREGROUND_SERVICE_DATA_SYNC)
        PermissionType.Notifications -> arrayOf(Manifest.permission.POST_NOTIFICATIONS)
    }
}
