/** Relative architectural display scale, independent of strategic travel distances. */
export const METRES_PER_MAP_UNIT = 8;
export const metres = (n: number) => n / METRES_PER_MAP_UNIT;
export const ROAD_WIDTH = metres(7);
export const DIRT_WIDTH = metres(5.5);
export const ROAD_SHOULDER = metres(0.65);
export const LANE_OFFSET = ROAD_WIDTH / 4;
export const MARKING_WIDTH = metres(0.12);
export const RAIL_GAUGE = metres(1.435);
export const TRAIN_LENGTH = metres(20);
export const TRAIN_SPACING = metres(21.5);
export const VEHICLE_DIMENSIONS = {
  compact: { length: metres(3.9), width: metres(1.75) },
  sedan: { length: metres(4.6), width: metres(1.85) },
  van: { length: metres(5.5), width: metres(2.05) },
  bus: { length: metres(11.5), width: metres(2.5) },
  truck: { length: metres(12), width: metres(2.5) },
} as const;
