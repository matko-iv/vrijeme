// Regression coverage for the real page renderer. Run: node test_daily_conditions.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, 'docs/forecast.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const context = vm.createContext({
    console, Date,
    document: {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, { textContent: '', innerHTML: '' });
            return elements.get(id);
        },
        querySelectorAll: () => [],
    },
    setInterval() {},
});
// Suppress network bootstrap and chart drawing; exercise the actual HTML renderer.
vm.runInContext(source.replace(/^loadForecast\(\);$/m, ''), context);
vm.runInContext('drawChart = () => {};', context);

const base = JSON.parse(fs.readFileSync(path.join(__dirname, 'docs/forecast_data/forecast_48h.json'), 'utf8'));
function renderDay(icon, description, narrative, options = {}) {
    const data = structuredClone(base);
    const day = Object.assign(data.daily_summary[0], {
        weather_icon: icon, weather_desc: description, day_narrative: narrative,
        precip_probability: 20, precip_total: 0.3, cloud_cover_day: 10,
    }, options);
    data.daily_summary = [day];
    data.long_range = [{ ...day, date: '2026-08-29' }];
    // Twelve sunny daytime hours and one afternoon storm used to erase the
    // published thunderstorm condition. Include actual hourly fields for render.
    data.hourly_forecast = Array.from({ length: 13 }, (_, i) => ({
        ...base.hourly_forecast[0], hour: i + 7,
        datetime: `${day.date}T${String(i + 7).padStart(2, '0')}:00:00`,
        weather_code: i === 8 ? 95 : 0,
    }));
    delete data.marine_forecast;
    const original = structuredClone(data);
    context.render(data);
    assert.deepEqual(data, original, 'render must not mutate published forecast data');
    return elements.get('app').innerHTML;
}

for (const [icon, description, narrative, file] of [
    ['thunderstorm', 'Grmljavina', 'Popodne stižu oblaci i grmljavina.', 'thunderstorms-day-rain.svg'],
    ['light_rain', 'Slaba kiša', 'Ujutro kratka kiša, zatim razvedravanje.', 'drizzle.svg'],
    ['snow', 'Snijeg', 'Kratkotrajan snijeg tokom večeri.', 'snow.svg'],
    ['clear', 'Vedro', 'Sunčano i toplo.', 'clear-day.svg'],
]) {
    const output = renderDay(icon, description, narrative);
    for (const [card, desc, text] of [
        ['day-card', 'weather-desc', 'day-narrative'],
        ['lr-card', 'lr-desc', 'lr-narrative'],
    ]) {
        const start = output.indexOf(`class="${card}"`);
        const section = output.slice(start, output.indexOf('class="temp-range"', start) > start
            ? output.indexOf('class="temp-range"', start) : undefined);
        assert.ok(section.includes(file), `${card}: ${icon} icon retained`);
        assert.ok(section.includes(`class="${desc}">${description}</div>`), `${card}: description retained`);
        assert.ok(section.includes(`class="${text}">${narrative}</div>`), `${card}: narrative retained`);
    }
}
const fallback = context.dailyCondition({ weather_code: 95 });
assert.equal(fallback.icon, 'thunderstorm');
assert.equal(fallback.description, 'Grmljavina');
assert.equal(context.dailyCondition({}).icon, 'partly_cloudy');
console.log('Daily and long-range conditions: all tests passed.');
