package ru.sr.core.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary = SRPalette.primary,
    onPrimary = SRPalette.onPrimary,
    primaryContainer = SRPalette.primaryContainer,
    onPrimaryContainer = SRPalette.onPrimaryContainer,
    secondary = SRPalette.secondary,
    onSecondary = SRPalette.onSecondary,
    background = SRPalette.background,
    onBackground = SRPalette.onBackground,
    surface = SRPalette.surface,
    onSurface = SRPalette.onSurface,
    tertiary = SRPalette.accent,
    onTertiary = SRPalette.onAccent
)

private val DarkColors = darkColorScheme(
    primary = SRPalette.primary,
    onPrimary = SRPalette.onPrimary,
    primaryContainer = SRPalette.primaryContainer,
    onPrimaryContainer = SRPalette.onPrimaryContainer,
    secondary = SRPalette.secondary,
    onSecondary = SRPalette.onSecondary,
    background = SRPalette.darkBackground,
    onBackground = SRPalette.onDarkBackground,
    surface = SRPalette.surfaceDark,
    onSurface = SRPalette.onSurfaceDark,
    tertiary = SRPalette.accent,
    onTertiary = SRPalette.onAccent
)

@Composable
fun SRTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColors else LightColors

    MaterialTheme(
        colorScheme = colorScheme,
        typography = SRTypography,
        content = content
    )
}
