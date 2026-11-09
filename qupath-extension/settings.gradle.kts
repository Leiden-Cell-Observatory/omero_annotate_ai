pluginManagement {
    repositories {
        gradlePluginPortal()
        maven {
            url = uri("https://maven.scijava.org/content/repositories/releases")
        }
    }
}

// The oldest QuPath version the extension supports (it is also tested with 0.7.0).
qupath {
    version = "0.6.0"
}

plugins {
    id("io.github.qupath.qupath-extension-settings") version "0.2.1"
}
