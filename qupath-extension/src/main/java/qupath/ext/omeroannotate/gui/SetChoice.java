package qupath.ext.omeroannotate.gui;

import qupath.ext.omeroannotate.core.Bundle;
import qupath.ext.omeroannotate.core.Row;
import qupath.ext.omeroannotate.core.RowCodec;
import qupath.ext.omeroannotate.core.SetConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * A training set as the chooser lists it: one with a folder on this computer, or one
 * found on OMERO that has no folder here yet.
 *
 * @param name     the set's name
 * @param folder   its folder on this computer, or null
 * @param tableId  its OMERO tracking table, or null for a set kept on this computer only
 * @param server   folder name of its OMERO server, or null
 * @param detail   one line under the name: where it is kept and how far it is
 */
public record SetChoice(String name, Path folder, Long tableId, String server, String detail) {

    public boolean isOnOmero() {
        return tableId != null;
    }

    /** The sets that have a folder under {@code home}, newest first. */
    public static List<SetChoice> local(Path home) {
        List<SetChoice> choices = new ArrayList<>();
        if (!Files.isDirectory(home))
            return choices;
        List<Path> configs = new ArrayList<>();
        try (Stream<Path> groups = Files.list(home)) {
            for (Path group : groups.filter(Files::isDirectory).toList()) {
                try (Stream<Path> bundles = Files.list(group)) {
                    bundles.map(b -> b.resolve(Bundle.CONFIG)).filter(Files::isRegularFile).forEach(configs::add);
                }
            }
        } catch (IOException e) {
            return choices;
        }
        configs.sort(Comparator.comparingLong((Path p) -> p.toFile().lastModified()).reversed());
        for (Path configFile : configs) {
            try {
                SetConfig config = SetConfig.load(configFile);
                Path folder = configFile.getParent();
                String progress = "";
                Path csv = folder.resolve(Bundle.TABLE);
                if (Files.exists(csv)) {
                    List<Row> rows = RowCodec.fromCsv(Files.readString(csv, StandardCharsets.UTF_8));
                    long done = rows.stream().filter(r -> r.processed).count();
                    progress = " · " + done + " of " + rows.size() + " done";
                }
                String where = config.isLocal() ? "On this computer"
                        : "On OMERO (" + (config.server() == null ? "?" : config.server()) + ", table " + config.tableId() + ")";
                String threeD = config.isThreeD() ? " · 3D (not for QuPath)" : "";
                choices.add(new SetChoice(config.name(), folder, config.isLocal() ? null : config.tableId(),
                        config.server(), where + progress + threeD));
            } catch (Exception e) {
                // a folder whose config cannot be read is not offered
            }
        }
        return choices;
    }
}
