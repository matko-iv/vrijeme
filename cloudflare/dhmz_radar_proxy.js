// Cloudflare Worker: serves DHMZ's Uljenje radar image with CORS headers, so
// docs/forecast.html can read its pixels in the browser. vrijeme.hr itself does
// not send Access-Control-Allow-Origin.
//
// Deploy: npx wrangler deploy cloudflare/dhmz_radar_proxy.js --name dhmz-radar
//             --compatibility-date 2026-10-01
// DHMZ_PROXY_URL in docs/forecast.html holds its URL. Nothing to run afterwards;
// the free plan covers 100k requests a day.

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
        // The page revalidates every 2 min: an unchanged frame costs a 304, not the PNG.
        const etag = upstream.headers.get('ETag');
        if (upstream.ok && etag && request.headers.get('If-None-Match') === etag) {
            return new Response(null, { status: 304, headers });
        }
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
