import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def http_get(url: str, params: dict | None = None, headers: dict | None = None) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "fog-care-finder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def geocode_location(query: str) -> tuple[float, float]:
    payload = {
        "q": query,
        "format": "json",
        "limit": 1,
    }
    raw = http_get(NOMINATIM_URL, payload)
    data = json.loads(raw.decode("utf-8"))
    if not data:
        raise ValueError("Location not found")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon


def overpass_hospitals(lat: float, lon: float, radius_m: int) -> list[dict]:
    # Amenities to consider: hospital, clinic, doctors, physiotherapist, pharmacy (optional)
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"^(hospital|clinic|doctors|physiotherapist)$"](around:{radius_m},{lat},{lon});
      way["amenity"~"^(hospital|clinic|doctors|physiotherapist)$"](around:{radius_m},{lat},{lon});
      relation["amenity"~"^(hospital|clinic|doctors|physiotherapist)$"](around:{radius_m},{lat},{lon});
    );
    out center tags 100;
    """
    data = http_get(OVERPASS_URL, params={"data": query})
    js = json.loads(data.decode("utf-8"))
    results: list[dict] = []
    for el in js.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("operator") or "Unnamed facility"
        phone = tags.get("phone") or tags.get("contact:phone")
        website = tags.get("website") or tags.get("contact:website")
        street = tags.get("addr:street")
        housenumber = tags.get("addr:housenumber")
        city = tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:suburb")
        postcode = tags.get("addr:postcode")
        addr_parts = [p for p in [street, housenumber] if p]
        line1 = " ".join(addr_parts) if addr_parts else None
        line2 = " ".join([p for p in [city, postcode] if p]) or None
        center = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        results.append({
            "name": name,
            "amenity": tags.get("amenity"),
            "phone": phone,
            "website": website,
            "address_line1": line1,
            "address_line2": line2,
            "lat": center.get("lat"),
            "lon": center.get("lon"),
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Find nearby hospitals/clinics via OpenStreetMap")
    ap.add_argument("location", type=str, help="Location text, e.g., 'Bengaluru, India'")
    ap.add_argument("--radius-km", type=float, default=10.0, help="Search radius in kilometers")
    ap.add_argument("--out", type=str, default="website/netlify/hospitals.json", help="Output JSON path")
    args = ap.parse_args()

    try:
        lat, lon = geocode_location(args.location)
        # Be polite: slight delay between services
        time.sleep(1.0)
        results = overpass_hospitals(lat, lon, int(args.radius_km * 1000))
        payload = {
            "query": args.location,
            "center": {"lat": lat, "lon": lon},
            "radius_km": args.radius_km,
            "count": len(results),
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} facilities to {args.out}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


