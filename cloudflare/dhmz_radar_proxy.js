// Cloudflare Worker: serves DHMZ's Uljenje radar image with CORS headers, so
// docs/forecast.html can read its pixels in the browser. Only needed if
// vrijeme.hr itself does not send Access-Control-Allow-Origin.
//
// Deploy: Cloudflare dashboard -> Workers & Pages -> Create -> Worker, paste this
// file, Deploy. Put the worker URL in DHMZ_PROXY_URL in docs/forecast.html.
// Nothing to run afterwards; the free plan covers 100k requests a day.

const SOURCE = 'https://vrijeme.hr/uljenje-stat.png';

export default {
    async fetch(request) {
        if (request.method === 'OPTIONS') {
            return new Response(null, { headers: corsHeaders() });
        }
        // Cache at the edge for a minute: DHMZ updates every 5-10 min, and every
        // visitor of the page then shares one upstream request.
        const upstream = await fetch(SOURCE, {
            headers: { 'User-Agent': 'budva-forecast/1.0 (radar status for one location)' },
            cf: { cacheTtl: 60, cacheEverything: true },
        });
        const headers = new Headers(corsHeaders());
        for (const name of ['Content-Type', 'Last-Modified', 'ETag']) {
            const value = upstream.headers.get(name);
            if (value) headers.set(name, value);
        }
        headers.set('Cache-Control', 'public, max-age=60');
        return new Response(upstream.body, { status: upstream.status, headers });
    },
};

function corsHeaders() {
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Expose-Headers': 'Last-Modified, ETag',
    };
}
