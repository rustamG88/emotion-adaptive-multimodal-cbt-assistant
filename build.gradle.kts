plugins {
    alias(libs.plugins.ktlint) apply false
    alias(libs.plugins.detekt) apply false
}

allprojects {
    apply(plugin = "org.jlleitschuh.gradle.ktlint")

    ktlint {
        android.set(true)
        ignoreFailures.set(false)
        filter {
            exclude("**/build/**")
        }
    }
}

subprojects {
    apply(plugin = "io.gitlab.arturbosch.detekt")

    detekt {
        toolVersion = libs.versions.detekt.get()
        config.setFrom(files("$rootDir/detekt.yml"))
        buildUponDefaultConfig = true
        autoCorrect = true
    }

    tasks.withType<io.gitlab.arturbosch.detekt.Detekt> {
        reports {
            html.required.set(true)
            xml.required.set(false)
            txt.required.set(false)
            sarif.required.set(false)
        }
    }
}
