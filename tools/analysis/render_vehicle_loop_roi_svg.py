import argparse
import html
import json
from pathlib import Path


PALETTE = (
    '#0072B2',
    '#D55E00',
    '#009E73',
    '#CC79A7',
    '#56B4E9',
    '#E69F00',
    '#C44E52',
    '#8172B2',
)

BACKGROUND = '#FFFFFF'
PLOT_BACKGROUND = '#F8F9FB'
PLOT_BORDER = '#D7DBE2'
TEXT_PRIMARY = '#22252B'
TEXT_SECONDARY = '#5C6370'
TOPOLOGY_LINE = '#C0C7D1'
ROI_STROKE = '#111111'
ROI_FILL = '#FFE082'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render ROI-specific vehicle loop routes on top of a scene topology subgraph.'
    )
    parser.add_argument('--loop-catalog', type=Path, required=True)
    parser.add_argument('--walkable-catalog', type=Path, required=True)
    parser.add_argument('--scene-id', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roi-margin', type=float, default=15.0)
    parser.add_argument('--single-pass-only', action='store_true')
    parser.add_argument('--max-routes', type=int, default=0, help='0 means render all matched routes.')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1100)
    return parser.parse_args()


def escape(text):
    return html.escape(str(text), quote=True)


def polyline_path(points):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)


def scene_cell(walkable_catalog, scene_id):
    catalog = json.loads(walkable_catalog.read_text(encoding='utf-8'))
    return next(cell for cell in catalog['cells'] if cell['id'] == scene_id)


def intersection_component_count(polyline, bounds):
    inside = []
    for x, y, _ in polyline:
        inside.append(
            bounds['min_x'] <= x <= bounds['max_x']
            and bounds['min_y'] <= y <= bounds['max_y']
        )

    components = 0
    prev = False
    for flag in inside:
        if flag and not prev:
            components += 1
        prev = flag
    return components


def filter_routes(loop_catalog, roi_bounds, single_pass_only, max_routes):
    catalog = json.loads(loop_catalog.read_text(encoding='utf-8'))
    matched = []
    for route in catalog['routes']:
        components = intersection_component_count(route['polyline'], roi_bounds)
        if components <= 0:
            continue
        if single_pass_only and components != 1:
            continue
        route_copy = dict(route)
        route_copy['roi_intersection_components'] = components
        matched.append(route_copy)

    matched.sort(key=lambda route: (-route['loop_length_m'], route['route_id']))
    if max_routes > 0:
        matched = matched[:max_routes]
    return catalog, matched


def gather_extent(cell, routes, roi_bounds):
    xs = []
    ys = []

    for edge in cell.get('edges', []):
        for x, y in edge.get('points', []):
            xs.append(float(x))
            ys.append(float(y))

    for route in routes:
        for x, y, _ in route['polyline']:
            xs.append(float(x))
            ys.append(float(y))

    xs.extend([roi_bounds['min_x'], roi_bounds['max_x']])
    ys.extend([roi_bounds['min_y'], roi_bounds['max_y']])

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    return {
        'min_x': min_x - span_x * 0.08,
        'max_x': max_x + span_x * 0.08,
        'min_y': min_y - span_y * 0.08,
        'max_y': max_y + span_y * 0.08,
    }


def make_projection(extent, width, height, margin=72, title_height=64):
    plot_left = margin
    plot_top = margin + title_height
    plot_width = width - margin * 2
    plot_height = height - plot_top - margin

    world_width = extent['max_x'] - extent['min_x']
    world_height = extent['max_y'] - extent['min_y']
    scale = min(plot_width / world_width, plot_height / world_height)

    used_width = world_width * scale
    used_height = world_height * scale
    offset_x = plot_left + (plot_width - used_width) * 0.5
    offset_y = plot_top + (plot_height - used_height) * 0.5

    def project(point):
        x, y = point
        canvas_x = offset_x + (x - extent['min_x']) * scale
        canvas_y = offset_y + used_height - (y - extent['min_y']) * scale
        return canvas_x, canvas_y

    return {
        'plot_left': plot_left,
        'plot_top': plot_top,
        'plot_width': plot_width,
        'plot_height': plot_height,
        'project': project,
    }


def render_svg(cell, matched_routes, roi_bounds, width, height, title, subtitle):
    extent = gather_extent(cell, matched_routes, roi_bounds)
    projection = make_projection(extent, width, height)
    project = projection['project']

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        f'<text x="{projection["plot_left"]}" y="42" font-family="Helvetica, Arial, sans-serif" font-size="28" font-weight="700" fill="{TEXT_PRIMARY}">{escape(title)}</text>',
        f'<text x="{projection["plot_left"]}" y="68" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="{TEXT_SECONDARY}">{escape(subtitle)}</text>',
        (
            f'<rect x="{projection["plot_left"]:.2f}" y="{projection["plot_top"]:.2f}" '
            f'width="{projection["plot_width"]:.2f}" height="{projection["plot_height"]:.2f}" '
            f'rx="20" fill="{PLOT_BACKGROUND}" stroke="{PLOT_BORDER}" stroke-width="1.5"/>'
        ),
    ]

    lines.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for edge in cell.get('edges', []):
        points = [(float(x), float(y)) for x, y in edge.get('points', [])]
        if len(points) < 2:
            continue
        projected = [project(point) for point in points]
        lines.append(
            f'<polyline points="{polyline_path(projected)}" stroke="{TOPOLOGY_LINE}" stroke-width="5.5" opacity="0.95"/>'
        )
    lines.append('</g>')

    roi_left_top = project((roi_bounds['min_x'], roi_bounds['max_y']))
    roi_right_bottom = project((roi_bounds['max_x'], roi_bounds['min_y']))
    roi_x = roi_left_top[0]
    roi_y = roi_left_top[1]
    roi_w = roi_right_bottom[0] - roi_left_top[0]
    roi_h = roi_right_bottom[1] - roi_left_top[1]
    lines.append(
        f'<rect x="{roi_x:.2f}" y="{roi_y:.2f}" width="{roi_w:.2f}" height="{roi_h:.2f}" '
        f'fill="{ROI_FILL}" opacity="0.10" stroke="{ROI_STROKE}" stroke-width="2.2" stroke-dasharray="10 8"/>'
    )

    lines.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for index, route in enumerate(matched_routes):
        color = PALETTE[index % len(PALETTE)]
        projected = [project((x, y)) for x, y, _ in route['polyline']]
        lines.append(
            f'<polyline points="{polyline_path(projected)}" stroke="{color}" stroke-width="2.6" opacity="0.68"/>'
        )
    lines.append('</g>')

    legend_x = projection['plot_left'] + 18
    legend_y = projection['plot_top'] + 20
    lines.append(
        f'<rect x="{legend_x - 8:.2f}" y="{legend_y - 16:.2f}" width="380" height="68" rx="12" fill="#FFFFFF" opacity="0.88" stroke="{PLOT_BORDER}" stroke-width="1"/>'
    )
    lines.append(
        f'<text x="{legend_x:.2f}" y="{legend_y:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="{TEXT_PRIMARY}">Gray: sidewalk_011 topology</text>'
    )
    lines.append(
        f'<text x="{legend_x:.2f}" y="{legend_y + 20:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="{TEXT_PRIMARY}">Dashed box: vehicle ROI (scene bounds + margin)</text>'
    )
    lines.append(
        f'<text x="{legend_x:.2f}" y="{legend_y + 40:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="{TEXT_PRIMARY}">Color: matched vehicle loop routes</text>'
    )

    lines.append('</svg>')
    return '\n'.join(lines)


def main():
    args = parse_args()
    cell = scene_cell(args.walkable_catalog, args.scene_id)
    bounds = cell['bounds']
    roi_bounds = {
        'min_x': float(bounds['min_x']) - args.roi_margin,
        'max_x': float(bounds['max_x']) + args.roi_margin,
        'min_y': float(bounds['min_y']) - args.roi_margin,
        'max_y': float(bounds['max_y']) + args.roi_margin,
    }

    catalog, matched_routes = filter_routes(
        args.loop_catalog,
        roi_bounds,
        single_pass_only=args.single_pass_only,
        max_routes=args.max_routes,
    )

    title = f'{catalog["map_name"]} {args.scene_id} vehicle loop routes'
    subtitle = (
        f'matched={len(matched_routes)} | '
        f'roi_margin={args.roi_margin:.1f}m | '
        f'filter={"single-pass" if args.single_pass_only else "all-pass"}'
    )
    svg = render_svg(cell, matched_routes, roi_bounds, args.width, args.height, title, subtitle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding='utf-8')
    print(f'matched_routes={len(matched_routes)}')
    print(f'svg={args.output}')


if __name__ == '__main__':
    main()
