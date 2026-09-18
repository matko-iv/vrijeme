import { Container, Sprite, Texture } from 'pixi.js';
import type { CityPlan } from './organic-cities';

// Architectural surfaces are drawn from the stable city plan, without AI art.
// Three sampling levels preserve the same geometry up to 24x on retina screens.
const EXTENT = 36;
let roofGrain: HTMLCanvasElement | undefined;
function roofMaterial(): HTMLCanvasElement {
    if (roofGrain) return roofGrain;
    const tile = document.createElement('canvas'); tile.width = tile.height = 128;
    const ctx = tile.getContext('2d', { willReadFrequently: true })!;
    let state = 901;
    for (let y = 0, row = 0; y < 128; y += 4, row++) {
        for (let x = -(row % 2) * 2; x < 128; x += 4) {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            ctx.fillStyle = state % 2 ? `rgba(241,215,166,${.03 + state % 17 / 100})` : `rgba(37,29,22,${.03 + state % 20 / 100})`;
            ctx.fillRect(x, y, 4, 4);
            ctx.fillStyle = '#e6cf9b30'; ctx.fillRect(x, y, 3, 1);
            ctx.fillStyle = '#201b193e'; ctx.fillRect(x, y + 3, 4, 1);
        }
    }
    roofGrain = tile; return tile;
}
export function citySurface(plan: CityPlan, detail: number): Container {
    const density = [8, 24, 64][detail];
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = EXTENT * density;
    const c = canvas.getContext('2d', { willReadFrequently: true })!;
    c.scale(density, density);
    c.translate(EXTENT / 2, EXTENT / 2);
    const roofPattern = c.createPattern(roofMaterial(), 'repeat')!;
    roofPattern.setTransform(new DOMMatrix().scale(1 / 72));
    let seed = 17321;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const polygon = (points: number[][], fill: string) => {
        c.beginPath(); points.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y));
        c.closePath(); c.fillStyle = fill; c.fill();
    };
    const disc = (x: number, y: number, r: number, color: string) => {
        c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fillStyle = color; c.fill();
    };
    // Irregular, translucent lots blend into the terrain instead of filling a hex.
    for (const b of plan.roofs) {
        const gradient = c.createRadialGradient(b.x, b.y, .3, b.x, b.y, 1.6);
        gradient.addColorStop(0, 'rgba(137,133,101,.78)');
        gradient.addColorStop(.55, 'rgba(124,127,87,.48)');
        gradient.addColorStop(1, 'rgba(116,125,79,0)');
        disc(b.x, b.y, 1.6, '#0000'); c.fillStyle = gradient; c.fill();
    }
    // Fine soil and vegetation mottling ties each plot to the aerial terrain.
    for (const b of plan.roofs) {
        for (let i = 0; i < 100; i++) {
            const a = random() * Math.PI * 2, r = Math.sqrt(random()) * 1.7;
            const alpha = (1 - r / 1.7) * .32;
            const light = 25 + random() * 23;
            disc(b.x + Math.cos(a) * r, b.y + Math.sin(a) * r, .015 + random() * .07, `hsla(${65 + random() * 20}, 23%, ${light}%, ${alpha})`);
        }
    }
    for (const p of plan.gardens) {
        c.save(); c.translate(p.x, p.y);
        c.fillStyle = '#64714c99'; c.fillRect(-.48, -.4, .96, .8);
        c.strokeStyle = '#b1aa8877'; c.lineWidth = .025; c.strokeRect(-.49, -.41, .98, .82);
        for (let i = 0; i < 100; i++) {
            c.fillStyle = random() > .5 ? '#c0b38d38' : '#263c2738';
            c.fillRect(random() * .94 - .47, random() * .78 - .39, .025, .024);
        }
        if (random() > .65) {
            c.strokeStyle = '#384b32aa'; c.lineWidth = .045;
            for (let x = -.32; x < .4; x += .13) { c.beginPath(); c.moveTo(x, -.28); c.lineTo(x, .28); c.stroke(); }
        }
        c.restore();
    }
    const streetPath = () => {
        c.beginPath();
        for (const line of plan.streets) line.forEach((p, i) => i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y));
    };
    c.lineCap = c.lineJoin = 'round';
    streetPath(); c.strokeStyle = '#393c3077'; c.lineWidth = .76; c.stroke();
    streetPath(); c.strokeStyle = '#928e7b'; c.lineWidth = .58; c.stroke();
    streetPath(); c.strokeStyle = '#65665d'; c.lineWidth = .48; c.stroke();
    // Asphalt aggregate, patches and pavement joints stay in world units.
    for (const line of plan.streets) {
        for (let i = 1; i < line.length; i++) {
            const a = line[i - 1], b = line[i], dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
            if (!len) continue;
            c.save(); c.translate(a.x, a.y); c.rotate(Math.atan2(dy, dx));
            for (let j = 0; j < len * 90; j++) {
                c.fillStyle = random() > .5 ? '#c9c5b42b' : '#242b282a';
                c.fillRect(random() * len, random() * .44 - .22, .014, .014);
            }
            c.restore();
        }
    }
    // Shadows are below every building, with consistent northwest sunlight.
    for (const b of plan.roofs) {
        c.save(); c.translate(b.x, b.y); c.rotate(b.angle);
        c.shadowColor = '#15221899'; c.shadowBlur = .09 * density;
        c.shadowOffsetX = .25 * density; c.shadowOffsetY = .38 * density;
        c.fillStyle = '#333a28'; c.fillRect(-b.width / 2, -b.height / 2, b.width, b.height);
        c.restore();
    }
    for (const b of plan.roofs) {
        c.save(); c.translate(b.x, b.y); c.rotate(b.angle);
        const w = b.width / 2, h = b.height / 2;
        const stone = ['#c4b99c', '#c6c0aa', '#aca998', '#d0bfa0'][b.tone];
        c.fillStyle = '#5e5d4b'; c.fillRect(-w + .035, -h + .07, 2 * w, 2 * h);
        c.fillStyle = stone; c.fillRect(-w, -h + .05, 2 * w, 2 * h);
        // Small facade openings are visible on the sunlit wall below the eave.
        c.fillStyle = '#303e36';
        for (let x = -w + .13; x < w - .06; x += .23) c.fillRect(x, h + .017, .07, .045);
        const hue = [26, 29, 42, 24][b.tone], sat = [27, 18, 7, 31][b.tone];
        const light = [47, 43, 49, 54][b.tone] + random() * 6 - 3;
        const hip = Math.min(w * .8, h * .36), ridge = h - hip;
        const facets = [
            { p: [[-w, -h], [0, -ridge], [0, ridge], [-w, h]], l: light + 8 },
            { p: [[0, -ridge], [w, -h], [w, h], [0, ridge]], l: light - 9 },
            { p: [[-w, -h], [w, -h], [0, -ridge]], l: light + 13 },
            { p: [[-w, h], [0, ridge], [w, h]], l: light - 3 }
        ];
        for (const f of facets) {
            polygon(f.p, `hsl(${hue} ${sat}% ${f.l}%)`);
            // Reuse a material tile rather than issuing thousands of tiny draws per roof.
            c.fillStyle = roofPattern; c.fill();
        }
        c.strokeStyle = '#d0b89599'; c.lineWidth = .023;
        c.beginPath(); c.moveTo(0, -ridge); c.lineTo(0, ridge);
        for (const [x, y, r] of [[-w, -h, -ridge], [w, -h, -ridge], [-w, h, ridge], [w, h, ridge]]) { c.moveTo(x, y); c.lineTo(0, r); }
        c.stroke();
        c.strokeStyle = '#292d2688'; c.lineWidth = .018; c.strokeRect(-w, -h, 2 * w, 2 * h);
        // Chimneys and occasional dormers add readable structure at street zoom.
        c.fillStyle = '#222b2777'; c.fillRect(w * .36 + .035, -h * .46 + .035, .11, .13);
        c.fillStyle = '#c3b494'; c.fillRect(w * .36, -h * .46, .10, .11);
        c.fillStyle = '#454337'; c.fillRect(w * .36 + .018, -h * .46 + .014, .063, .054);
        if (b.tone === 2) {
            c.fillStyle = '#263f4966'; c.fillRect(-w * .8, -h * .3, w * .48, h * .57);
            c.strokeStyle = '#c2c0a67a'; c.lineWidth = .017; c.strokeRect(-w * .8, -h * .3, w * .48, h * .57);
        }
        c.restore();
    }
    for (const p of plan.trees) {
        const size = .25 + random() * .31;
        const shadow = c.createRadialGradient(p.x + .23, p.y + .3, .1, p.x + .23, p.y + .3, size * 1.2);
        shadow.addColorStop(0, '#182218aa'); shadow.addColorStop(.65, '#18221870'); shadow.addColorStop(1, '#18221800');
        disc(p.x + .23, p.y + .3, size * 1.2, '#0000'); c.fillStyle = shadow; c.fill();
        disc(p.x, p.y, size, '#34472c');
        for (let i = 0; i < 100; i++) {
            const a = random() * 6.283, r = Math.sqrt(random()) * size * .85;
            const x = Math.cos(a) * r, y = Math.sin(a) * r;
            const l = 25 + random() * 11 - (x + y) * 15;
            disc(p.x + x, p.y + y, size * (.07 + random() * .15), `hsl(${72 + random() * 23} 24% ${l}%)`);
        }
    }
    const texture = Texture.from(canvas);
    texture.source.scaleMode = 'linear';
    const sprite = new Sprite(texture); sprite.anchor.set(.5); sprite.width = sprite.height = EXTENT;
    const container = new Container(); container.addChild(sprite); return container;
}
