"""Apply narrow, asserted edits to the staged copy of the game's renderer."""
from pathlib import Path
root=Path(__file__).parent/'frontend/src/lib/game'
def replace(s,a,b):
    assert a in s,a[:120]
    return s.replace(a,b)
p=root/'map-renderer.ts';s=p.read_text(encoding='utf-8')
s=replace(s,"import { TrafficLayer } from './traffic-layer';","import { TrafficLayer } from './traffic-layer';\nimport { OrganicTerrain } from './organic-terrain';\nimport { OrganicCities } from './organic-cities';\nimport { ROAD_WIDTH, DIRT_WIDTH, ROAD_SHOULDER, MARKING_WIDTH, RAIL_GAUGE, metres } from './map-scale';")
s=replace(s,'private maxZoom = 4.0;','private maxZoom = 24.0;')
s=replace(s,'private terrainTexturesLoaded = false;','private terrainTexturesLoaded = false;\n  private organic = new OrganicTerrain();\n  private cities = new OrganicCities();\n  private terrainLoading = false;\n  private disposed = false;')
s=replace(s,'this.worldContainer.addChild(this.tileContainer);','this.worldContainer.addChild(this.organic.container);\n    this.worldContainer.addChild(this.tileContainer);\n    this.worldContainer.addChild(this.cities.container);')
s=replace(s,'if (changed) this.dirty = true;','if (changed) { this.dirty = true; this.organic.dirty = true; }')
s=replace(s,'this.transport = new TransportNetwork(this.tiles, this.networkEdges);','this.organic.dirty = true;\n    this.cities.clear();\n    this.transport = new TransportNetwork(this.tiles, this.networkEdges);')
s=replace(s,'this.lakeHexes.clear();','this.organic.dirty = true;\n    this.cities.clear();\n    this.lakeHexes.clear();')
start=s.index('  private async preloadTerrainTileTextures()')
end=s.index('\n  private ',start+10)
s=s[:start]+'''  private async preloadTerrainTileTextures() {
    if (this.terrainLoading || this.terrainTexturesLoaded || this.disposed) return;
    this.terrainLoading = true;
    try {
      await this.organic.load();
      if (!this.disposed) { this.terrainTexturesLoaded = true; this.dirty = true; }
    } catch (error) {
      // Flat colours remain usable while the normal texture fallback loads.
      console.warn('[map-renderer] Organic imagery unavailable, using fallback', error);
      try {
        await preloadTerrainTextures(new Set([...this.tiles.values()].map(t => t.terrain)));
        if (!this.disposed) { this.terrainTexturesLoaded = areTerrainTexturesLoaded(); this.dirty = true; }
      } catch (fallbackError) { console.warn('[map-renderer] Texture fallback unavailable', fallbackError); }
    } finally { this.terrainLoading = false; }
  }
''' +s[end:]
s=replace(s,'  destroy() {','  destroy() {\n    this.disposed = true;\n    this.organic.destroy();\n    this.cities.destroy();')
s=replace(s,'    this.tileContainer.removeChildren();\n    this.tileContainer.visible = useTextures;','''    const useOrganic = useTextures && this.organic.ready;
    this.organic.setWorld(this.tiles, this.lakeHexes, this.snowHexes, this.season === 'winter');
    this.organic.setView(offsetX, offsetY, z, w, h, useOrganic);
    this.cities.draw(vis, visCount, this.transport, z, useOrganic, this.lakeHexes);
    this.tileContainer.removeChildren();
    this.tileContainer.visible = useTextures && !useOrganic;''')
s=replace(s,'this.baseGfx.visible = useTextures;','this.baseGfx.visible = useTextures && !useOrganic;')
s=replace(s,'        // Priority: city settlement > installation > terrain','        if (!useOrganic) {\n        // Legacy fallback: city settlement > installation > terrain')
s=replace(s,'        // Political tint over the artwork so ownership reads clearly','        }\n        // Political tint over the artwork so ownership reads clearly')
s=replace(s,'if (drawBorders) borderPolys.push(corners());','if (drawBorders && (!useOrganic || this._showBorders)) borderPolys.push(corners());')
s=replace(s,'const roadWidth = Math.max(1.5, z * 2.0);','const roadWidth = ROAD_WIDTH * z;')
s=replace(s,'const railWidth = Math.max(0.6, z * 0.7);','const railWidth = metres(.09) * z;')
s=replace(s,'const tieLength = Math.max(1.2, z * 2.0);','const tieLength = metres(2.6) / 2 * z;')
s=replace(s,'const width = roadWidth * (dirt ? .72 : busy ? 1.2 : 1);','const width = dirt ? DIRT_WIDTH * z : roadWidth;')
s=replace(s,'width + Math.max(.6, z * .65)','width + ROAD_SHOULDER * 2 * z')
s=replace(s,'dirt ? 0x8d7955 : 0x353b3d','dirt ? 0x8d7955 : 0x989184')
s=replace(s,'busy ? 0x69717a : 0x73787b','busy ? 0x515755 : 0x60635d')
s=replace(s,'for (const lane of [-.4, .4])','for (const lane of [-DIRT_WIDTH/4, DIRT_WIDTH/4])')
s=replace(s,'width: z * .22','width: z * metres(.28)')
s=replace(s,'if (z > 1.1 && busy)','if (z > 3 && !dirt)')
s=replace(s,'d += 4','d += metres(9)')
s=replace(s,'d + 1.8','d + metres(3)')
s=replace(s,'color: 0xefdcaa, width: Math.max(.3, z * .25)','color: 0xd9d7c7, width: MARKING_WIDTH * z')
s=replace(s,"const half = path.surface === 'dirt' ? .72 : path.activity > .55 ? 1.2 : 1;","const half = (path.surface === 'dirt' ? DIRT_WIDTH : ROAD_WIDTH) / 2;")
s=replace(s,'if (!dirt && z > 1.2)','if (!dirt && z > 3)')
s=replace(s,'(p.x - nx) * z','(p.x - nx * ROAD_WIDTH / 2) * z')
s=replace(s,'(p.y - ny) * z','(p.y - ny * ROAD_WIDTH / 2) * z')
s=replace(s,'(p.x + nx) * z','(p.x + nx * ROAD_WIDTH / 2) * z')
s=replace(s,'(p.y + ny) * z','(p.y + ny * ROAD_WIDTH / 2) * z')
s=replace(s,'width: z * .3','width: z * MARKING_WIDTH * 2')
s=replace(s,'d += 3.5','d += metres(.65)')
s=replace(s,'if (z < .85)','if (z < 4)')
s=replace(s,'width: Math.max(.4, z * .55)','width: metres(.20) * z')
s=replace(s,'side * .8, false','side * RAIL_GAUGE / 2, false')
p.write_text(s,encoding='utf-8')
p=root/'transport.ts';s=p.read_text(encoding='utf-8')
s=replace(s,"import { fitRoutes } from './route-geometry';","import { fitRoutes } from './route-geometry';\nimport { VEHICLE_DIMENSIONS, LANE_OFFSET } from './map-scale';")
s=replace(s,"({compact:2.6,sedan:3.2,van:4,bus:5.3,truck:5.6})[v.model ?? 'sedan']","VEHICLE_DIMENSIONS[v.model ?? 'sedan'].length")
s=replace(s,"path.kind === 'road' ? .65 : 0","path.kind === 'road' ? LANE_OFFSET : 0")
s=replace(s,'[0xf1d29b, 0xe5e8ed, 0xcf654c, 0x69a1b5, 0x828db2]','[0xd9d9d2, 0xa7adae, 0x753f36, 0x3c5361, 0x454b4b]')
p.write_text(s,encoding='utf-8')
