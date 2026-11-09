plugins {
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-omero-annotate-ai"
    group = "io.github.leiden-cell-observatory"
    version = "0.1.0"
    description = "Annotate omero-annotate-ai training sets in QuPath (OMERO or local)"
    automaticModule = "io.github.leidencellobservatory.extension.omeroannotate"
}

dependencies {
    // Provided by QuPath itself.
    compileOnly(libs.bundles.qupath)
    compileOnly(libs.bundles.logging)
    compileOnly(libs.qupath.fxtras)
    compileOnly(libs.snakeyaml)
    compileOnly(libs.gson)

    // Optional at run time: the OMERO Java gateway (tracking tables need it). It is not
    // bundled. The official OMERO extension is reached by reflection (to reuse its login).
    compileOnly("org.openmicroscopy:omero-gateway:5.10.3")

    testImplementation(libs.bundles.qupath)
    testImplementation(libs.snakeyaml)
    testImplementation(libs.gson)
    testImplementation(libs.junit)
    testImplementation("org.openmicroscopy:omero-gateway:5.10.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

repositories {
    maven {
        name = "ome.maven"
        url = uri("https://artifacts.openmicroscopy.org/artifactory/maven")
    }
    maven {
        name = "SciJava Public"
        url = uri("https://maven.scijava.org/content/repositories/public/")
    }
}

tasks.test {
    useJUnitPlatform()
    // Tests tagged "omero" need a running OMERO server (see README).
    systemProperty("omero.test.host", System.getProperty("omero.test.host") ?: "")
}
