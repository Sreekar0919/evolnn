import pygame


def ray_rectangle_collision(
    start: pygame.Vector2,
    direction: pygame.Vector2,
    rect: pygame.Rect,
    max_length: float,
) -> float:
    """Return the distance from a ray origin to the nearest rectangle edge."""
    corners = [pygame.Vector2(point) for point in rect.corners]
    edges = zip(corners, corners[1:] + corners[:1])
    closest_distance = max_length

    for start_point, end_point in edges:
        edge = end_point - start_point
        denominator = direction.cross(edge)
        if abs(denominator) < 0.00001:
            continue

        difference = start_point - start
        distance = difference.cross(edge) / denominator
        position = difference.cross(direction) / denominator
        if distance >= 0 and 0 <= position <= 1:
            closest_distance = min(closest_distance, distance)

    return closest_distance


def ray_mask_collision(
    start: pygame.Vector2,
    direction: pygame.Vector2,
    mask: pygame.Mask,
    bounds: pygame.Rect,
    max_length: int,
    step: int = 2,
) -> tuple[float, pygame.Vector2]:
    """Return the first occupied mask pixel reached by a ray."""
    for distance in range(0, max_length + 1, step):
        point = start + direction * distance
        pixel = round(point.x), round(point.y)
        if not bounds.collidepoint(pixel) or mask.get_at(pixel):
            return distance, point

    return max_length, start + direction * max_length