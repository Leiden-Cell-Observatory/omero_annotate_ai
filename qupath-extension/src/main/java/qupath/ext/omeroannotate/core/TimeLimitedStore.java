package qupath.ext.omeroannotate.core;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Gives every call to OMERO a time limit.
 * <p>
 * A server that is switched off refuses a call at once, but one that is frozen or behind
 * a broken network link simply never answers, and the annotator would wait for ever. With
 * a limit such a call fails like any other, the mask stays safe in the set's folder and
 * is uploaded later. A call that does finish after its limit is harmless: uploads carry
 * the unit's tag and are adopted, never repeated.
 */
public final class TimeLimitedStore {

    /** Calls that never wait for the server. */
    private static final Set<String> LOCAL = Set.of("close", "serverSlug", "userName", "toString", "hashCode", "equals");

    /**
     * Calls that send or remove data and may rightly take long (thousands of ROIs, a
     * large table). They get more time, because giving up on one that is still running
     * is worse than waiting.
     */
    private static final Set<String> SLOW = Set.of("uploadLabel", "uploadRois", "delete", "createTable",
            "replaceConfig", "writeRows");
    private static final long SLOW_FACTOR = 5;

    private static final ExecutorService CALLS = Executors.newCachedThreadPool(r -> {
        Thread thread = new Thread(r, "omero-annotate-ai-call");
        thread.setDaemon(true);
        return thread;
    });

    /** How long after an unanswered call further calls fail at once. */
    private static final long QUIET_NANOS = TimeUnit.SECONDS.toNanos(8);

    private TimeLimitedStore() {}

    /** {@code store}, with every server call failing after {@code seconds} without an answer. */
    public static RemoteStore wrap(RemoteStore store, long seconds) {
        long[] lastTimeout = {0};
        return (RemoteStore) Proxy.newProxyInstance(RemoteStore.class.getClassLoader(),
                new Class<?>[]{RemoteStore.class}, (proxy, method, args) -> {
                    if (LOCAL.contains(method.getName())) {
                        try {
                            return method.invoke(store, args);
                        } catch (InvocationTargetException e) {
                            throw e.getCause();
                        }
                    }
                    // Right after a call went unanswered the next ones are not tried, so a
                    // silent server costs one wait, not one per step.
                    if (System.nanoTime() - lastTimeout[0] < QUIET_NANOS && lastTimeout[0] != 0)
                        throw new IOException("The OMERO server is not answering");
                    Future<Object> call = CALLS.submit(() -> method.invoke(store, args));
                    try {
                        return call.get(SLOW.contains(method.getName()) ? seconds * SLOW_FACTOR : seconds,
                                TimeUnit.SECONDS);
                    } catch (TimeoutException e) {
                        call.cancel(true);
                        lastTimeout[0] = System.nanoTime();
                        throw new IOException("The OMERO server did not answer within "
                                + (SLOW.contains(method.getName()) ? seconds * SLOW_FACTOR : seconds) + " seconds");
                    } catch (ExecutionException e) {
                        Throwable cause = e.getCause();
                        throw cause instanceof InvocationTargetException target ? target.getCause() : cause;
                    }
                });
    }
}
