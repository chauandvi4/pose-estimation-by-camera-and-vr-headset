"""Simple OSC receiver that ingests Unity upper-body pose packets.

This utility listens for the JSON payload emitted by ``OscPoseSender`` and
prints the decoded Quest/upper-body pose data. It is intended to provide a
lightweight bridge between Unity and the Python fusion pipeline so that the
communication layer can be validated without a connected headset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


class OscError(RuntimeError):
    """Raised when an OSC message cannot be decoded."""


@dataclass
class OscMessage:
    address: str
    arguments: Sequence[str]


def _read_osc_string(buffer: bytes, offset: int) -> Tuple[str, int]:
    try:
        end = buffer.index(0, offset)
    except ValueError as exc:  # pragma: no cover - defensive; datagrams are tiny
        raise OscError("OSC string is not null terminated") from exc

    value = buffer[offset:end].decode("ascii")
    next_index = end + 1
    next_index = (next_index + 3) & ~0x03
    return value, min(next_index, len(buffer))


def parse_osc_message(buffer: bytes) -> OscMessage:
    """Parse the subset of OSC used by :class:`OscPoseSender`."""

    address, index = _read_osc_string(buffer, 0)
    type_tags, index = _read_osc_string(buffer, index)

    if not type_tags.startswith(","):
        raise OscError(f"Invalid OSC type tag string: {type_tags!r}")

    arguments: List[str] = []
    for tag in type_tags[1:]:
        if tag == "s":
            value, index = _read_osc_string(buffer, index)
            arguments.append(value)
        else:
            raise OscError(f"Unsupported OSC type tag: {tag!r}")

    return OscMessage(address=address, arguments=tuple(arguments))


class PosePacketProtocol(asyncio.DatagramProtocol):
    def __init__(self, handler: Callable[[dict, Tuple[str, int]], None]) -> None:
        super().__init__()
        self._handler = handler

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            message = parse_osc_message(data)
            if not message.arguments:
                logger.debug("OSC message %s had no arguments", message.address)
                return
            payload = json.loads(message.arguments[0])
        except (OscError, json.JSONDecodeError) as exc:
            logger.warning("Failed to decode OSC packet from %s: %s", addr, exc)
            return

        self._handler(payload, addr)


async def run_server(host: str, port: int, on_packet: Callable[[dict, Tuple[str, int]], None]) -> None:
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: PosePacketProtocol(on_packet), local_addr=(host, port)
    )

    logger.info("Listening for OSC pose packets on udp://%s:%d", host, port)
    try:
        await asyncio.Future()
    finally:
        transport.close()


def _pretty_print_packet(packet: dict, addr: Tuple[str, int]) -> None:
    timestamp = packet.get("timestamp")
    hmd = packet.get("hmd", {})
    position = hmd.get("position", {})
    rotation = hmd.get("rotation", {})

    logger.info(
        "Packet from %s timestamp=%s hmd=(%.3f, %.3f, %.3f) yaw=%.1f",
        addr,
        timestamp,
        position.get("x", 0.0),
        position.get("y", 0.0),
        position.get("z", 0.0),
        rotation.get("y", 0.0),
    )

    joint_iterable: Iterable[dict] = packet.get("joints", [])
    joint_list = list(joint_iterable)
    if joint_list:
        joint_summary = ", ".join(
            f"{joint.get('name')}:({joint.get('pose', {}).get('position', {}).get('x', 0.0):.3f},"
            f" {joint.get('pose', {}).get('position', {}).get('y', 0.0):.3f},"
            f" {joint.get('pose', {}).get('position', {}).get('z', 0.0):.3f})"
            for joint in joint_list[:4]
        )
        if len(joint_list) > 4:
            joint_summary += ", ..."
        logger.debug("Sample joints %s", joint_summary)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on (default: 9000)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity (prints sample joint positions)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        asyncio.run(run_server(args.host, args.port, _pretty_print_packet))
    except KeyboardInterrupt:
        logger.info("Shutting down OSC receiver")


if __name__ == "__main__":
    main()
