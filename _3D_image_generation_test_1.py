#!/usr/bin/env python3
"""Render a slowly spinning, polished 3D wedding band with Matplotlib.

Requirements: numpy, matplotlib, and FFmpeg installed and available on PATH.
Run: python3 wedding_ring.py --output wedding_ring.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np


# A softly rounded, classic band. R is the band centerline radius; the two
# smaller radii control radial thickness and band width, respectively.
R = 1.18
RADIAL_THICKNESS = 0.145
HALF_WIDTH = 0.205


def ring_mesh(theta_count: int = 180, phi_count: int = 72):
    theta = np.linspace(0, 2 * np.pi, theta_count, endpoint=True)
    phi = np.linspace(0, 2 * np.pi, phi_count, endpoint=True)
    theta, phi = np.meshgrid(theta, phi)
    radial = R + RADIAL_THICKNESS * np.cos(phi)
    x = radial * np.cos(theta)
    y = radial * np.sin(theta)
    z = HALF_WIDTH * np.sin(phi)

    # Outward normals for the gently elliptical torus section.
    nx = np.cos(phi) * np.cos(theta)
    ny = np.cos(phi) * np.sin(theta)
    nz = (RADIAL_THICKNESS / HALF_WIDTH) * np.sin(phi)
    normals = np.stack((nx, ny, nz), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)
    return np.stack((x, y, z), axis=-1), normals


def rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gold_shading(points: np.ndarray, normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Approximate polished metallic reflections with diffuse and specular lights."""
    world = points @ rotation.T
    n = normals @ rotation.T
    camera = np.array([4.2, -6.5, 3.7], dtype=float)
    view = camera - world
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-9)

    # Warm ambient fill plus large softbox and narrower strip-box highlights.
    lights = [
        (np.array([-0.42, -0.58, 0.70]), np.array([1.00, 0.91, 0.69]), 0.95, 28),
        (np.array([0.66, 0.24, 0.71]), np.array([0.76, 0.86, 1.00]), 0.68, 42),
        (np.array([-0.16, 0.96, 0.22]), np.array([1.00, 0.78, 0.44]), 0.45, 76),
    ]
    base = np.array([0.72, 0.355, 0.075])
    rgb = np.broadcast_to(base * 0.19, normals.shape).copy()
    for direction, tint, strength, power in lights:
        direction = direction / np.linalg.norm(direction)
        ndotl = np.maximum(np.sum(n * direction, axis=-1, keepdims=True), 0.0)
        half_vector = direction + view
        half_vector /= np.maximum(np.linalg.norm(half_vector, axis=-1, keepdims=True), 1e-9)
        spec = np.maximum(np.sum(n * half_vector, axis=-1, keepdims=True), 0.0) ** power
        diffuse = 0.23 * ndotl * base
        metallic_spec = 0.92 * spec * tint
        rgb += strength * (diffuse + metallic_spec)

    # Gentle contrast rolloff followed by linear-to-sRGB conversion keeps the
    # gold bright on screen while retaining warm shadows and controlled glints.
    rgb = rgb / (0.78 + rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0.025, 1.0)


def render(output: Path, duration: float, fps: int, size: int, dpi: int) -> None:
    points, normals = ring_mesh()
    frames = int(round(duration * fps))
    output.parent.mkdir(parents=True, exist_ok=True)

    bg = "#10141a"
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111, projection="3d", facecolor=bg)
    ax.set_position([0.015, 0.015, 0.97, 0.97])
    ax.set_xlim(-1.82, 1.82)
    ax.set_ylim(-1.82, 1.82)
    ax.set_zlim(-1.58, 1.58)
    ax.set_box_aspect((1, 1, 0.87))
    ax.view_init(elev=21, azim=-53)
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(bg)
        axis.line.set_color(bg)

    surface = [None]

    def update(frame: int):
        if surface[0] is not None:
            surface[0].remove()
        # One full gentle rotation per clip, with a small initial tilt that
        # makes the band's interior and rounded profile visible throughout.
        progress = frame / frames
        spin = 2 * np.pi * progress
        spin_matrix = rotation_y(0.25 + spin)
        initial_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(0.12), -np.sin(0.12)],
            [0, np.sin(0.12), np.cos(0.12)],
        ])
        transform = spin_matrix @ initial_tilt
        transformed = points @ transform.T
        colors = gold_shading(points, normals, transform)
        surface[0] = ax.plot_surface(
            transformed[..., 0], transformed[..., 1], transformed[..., 2],
            rstride=1, cstride=1, facecolors=colors,
            linewidth=0, antialiased=True, shade=False,
        )
        return (surface[0],)

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=5200, codec="h264", extra_args=["-pix_fmt", "yuv420p"])
    animation.save(output, writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("wedding_ring.mp4"))
    parser.add_argument("--duration", type=float, default=8.0, help="seconds (default: 8)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--size", type=int, default=720, help="square output pixels (default: 720)")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if args.duration <= 0 or args.fps <= 0 or args.size < 240:
        parser.error("duration and fps must be positive; size must be at least 240")
    render(args.output, args.duration, args.fps, args.size, args.dpi)
    print(f"Saved {args.output} ({args.duration:g}s, {args.fps} fps, {args.size}x{args.size})")


if __name__ == "__main__":
    main()

