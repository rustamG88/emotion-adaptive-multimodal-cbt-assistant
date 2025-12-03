"""
🎮 ТАНКИ - Классическая аркадная игра на Python
Автор: AI Assistant
Управление:
  - WASD или стрелки: движение танка
  - ПРОБЕЛ: стрельба
  - R: рестарт после проигрыша
  - ESC: выход
"""

import pygame
import random
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Инициализация pygame
pygame.init()
pygame.mixer.init()

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

# Размеры экрана
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
TILE_SIZE = 48

# Цвета (военная палитра)
COLORS = {
    'bg': (28, 35, 28),              # Тёмно-зелёный фон
    'grid': (35, 45, 35),            # Сетка
    'player': (76, 153, 76),         # Зелёный танк игрока
    'player_dark': (51, 102, 51),    # Тёмный оттенок
    'enemy': (179, 89, 89),          # Красный танк врага
    'enemy_dark': (128, 64, 64),     # Тёмный оттенок
    'bullet_player': (255, 230, 150),# Жёлтый снаряд
    'bullet_enemy': (255, 140, 140), # Красный снаряд
    'wall': (139, 119, 101),         # Кирпичная стена
    'wall_dark': (101, 86, 73),      # Тёмный кирпич
    'steel': (140, 150, 160),        # Металл
    'steel_light': (180, 190, 200),  # Светлый металл
    'water': (64, 128, 200),         # Вода
    'water_light': (100, 170, 255),  # Светлая вода
    'grass': (50, 90, 50),           # Трава
    'base': (255, 200, 50),          # База (орёл)
    'explosion': (255, 180, 50),     # Взрыв
    'text': (220, 220, 200),         # Текст
    'text_shadow': (30, 30, 30),     # Тень текста
    'health_bar': (100, 200, 100),   # Полоска здоровья
    'health_bg': (60, 60, 60),       # Фон полоски здоровья
    'hud_bg': (20, 25, 20, 200),     # Фон HUD
}

# Направления
class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# Скорости
PLAYER_SPEED = 4
ENEMY_SPEED = 2
BULLET_SPEED = 8
ENEMY_BULLET_SPEED = 6

# FPS
FPS = 60

# =============================================================================
# ИГРОВЫЕ ОБЪЕКТЫ
# =============================================================================

@dataclass
class Vector2:
    """2D вектор"""
    x: float
    y: float
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)


class Bullet:
    """Снаряд танка"""
    
    def __init__(self, x: float, y: float, direction: Direction, is_player: bool):
        self.x = x
        self.y = y
        self.direction = direction
        self.is_player = is_player
        self.speed = BULLET_SPEED if is_player else ENEMY_BULLET_SPEED
        self.radius = 6
        self.alive = True
        self.trail = []  # След от снаряда
        
    def update(self):
        # Сохраняем позицию для следа
        self.trail.append((self.x, self.y))
        if len(self.trail) > 5:
            self.trail.pop(0)
            
        # Движение
        dx, dy = self._get_velocity()
        self.x += dx
        self.y += dy
        
        # Проверка границ
        if (self.x < 0 or self.x > SCREEN_WIDTH or 
            self.y < 0 or self.y > SCREEN_HEIGHT):
            self.alive = False
            
    def _get_velocity(self) -> Tuple[float, float]:
        if self.direction == Direction.UP:
            return (0, -self.speed)
        elif self.direction == Direction.DOWN:
            return (0, self.speed)
        elif self.direction == Direction.LEFT:
            return (-self.speed, 0)
        else:
            return (self.speed, 0)
    
    def draw(self, screen: pygame.Surface):
        # След
        for i, pos in enumerate(self.trail):
            alpha = int(100 * (i + 1) / len(self.trail))
            color = COLORS['bullet_player'] if self.is_player else COLORS['bullet_enemy']
            faded = tuple(int(c * alpha / 255) for c in color)
            pygame.draw.circle(screen, faded, (int(pos[0]), int(pos[1])), 3)
        
        # Снаряд
        color = COLORS['bullet_player'] if self.is_player else COLORS['bullet_enemy']
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        
        # Свечение
        glow_surf = pygame.Surface((self.radius * 4, self.radius * 4), pygame.SRCALPHA)
        for i in range(3):
            alpha = 50 - i * 15
            pygame.draw.circle(glow_surf, (*color[:3], alpha), 
                             (self.radius * 2, self.radius * 2), self.radius + i * 3)
        screen.blit(glow_surf, (int(self.x) - self.radius * 2, int(self.y) - self.radius * 2))
    
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)


class Tank:
    """Базовый класс танка"""
    
    def __init__(self, x: float, y: float, is_player: bool = False):
        self.x = x
        self.y = y
        self.is_player = is_player
        self.direction = Direction.UP
        self.speed = PLAYER_SPEED if is_player else ENEMY_SPEED
        self.size = 40
        self.health = 3 if is_player else 1
        self.max_health = self.health
        self.alive = True
        self.shoot_cooldown = 0
        self.shoot_delay = 20 if is_player else 60
        self.invincible_timer = 60 if is_player else 0  # Неуязвимость при спавне
        
        # AI для врагов
        self.ai_timer = 0
        self.ai_move_time = random.randint(30, 90)
        
    def update(self, walls: List[pygame.Rect]):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.invincible_timer > 0:
            self.invincible_timer -= 1
            
    def move(self, direction: Direction, walls: List[pygame.Rect]):
        self.direction = direction
        dx, dy = 0, 0
        
        if direction == Direction.UP:
            dy = -self.speed
        elif direction == Direction.DOWN:
            dy = self.speed
        elif direction == Direction.LEFT:
            dx = -self.speed
        elif direction == Direction.RIGHT:
            dx = self.speed
            
        # Проверка коллизий
        new_x = self.x + dx
        new_y = self.y + dy
        new_rect = pygame.Rect(new_x - self.size//2, new_y - self.size//2, 
                               self.size, self.size)
        
        # Границы экрана
        if new_x - self.size//2 < 0:
            new_x = self.size//2
        if new_x + self.size//2 > SCREEN_WIDTH:
            new_x = SCREEN_WIDTH - self.size//2
        if new_y - self.size//2 < 0:
            new_y = self.size//2
        if new_y + self.size//2 > SCREEN_HEIGHT:
            new_y = SCREEN_HEIGHT - self.size//2
            
        new_rect = pygame.Rect(new_x - self.size//2, new_y - self.size//2, 
                               self.size, self.size)
        
        # Проверка столкновений со стенами
        can_move = True
        for wall in walls:
            if new_rect.colliderect(wall):
                can_move = False
                break
                
        if can_move:
            self.x = new_x
            self.y = new_y
            
        return can_move
    
    def shoot(self) -> Optional[Bullet]:
        if self.shoot_cooldown <= 0:
            self.shoot_cooldown = self.shoot_delay
            
            # Позиция снаряда перед дулом
            offset = self.size // 2 + 10
            bx, by = self.x, self.y
            
            if self.direction == Direction.UP:
                by -= offset
            elif self.direction == Direction.DOWN:
                by += offset
            elif self.direction == Direction.LEFT:
                bx -= offset
            elif self.direction == Direction.RIGHT:
                bx += offset
                
            return Bullet(bx, by, self.direction, self.is_player)
        return None
    
    def take_damage(self) -> bool:
        """Получение урона. Возвращает True если танк уничтожен."""
        if self.invincible_timer > 0:
            return False
        self.health -= 1
        if self.health <= 0:
            self.alive = False
            return True
        self.invincible_timer = 30
        return False
    
    def draw(self, screen: pygame.Surface):
        # Цвета
        if self.is_player:
            main_color = COLORS['player']
            dark_color = COLORS['player_dark']
        else:
            main_color = COLORS['enemy']
            dark_color = COLORS['enemy_dark']
            
        # Эффект неуязвимости - мигание
        if self.invincible_timer > 0 and self.invincible_timer % 6 < 3:
            main_color = tuple(min(255, c + 80) for c in main_color)
        
        # Корпус танка
        body_rect = pygame.Rect(self.x - self.size//2, self.y - self.size//2, 
                                self.size, self.size)
        
        # Вращаем и рисуем танк
        self._draw_tank_body(screen, body_rect, main_color, dark_color)
        
        # Полоска здоровья (только для игрока)
        if self.is_player and self.health < self.max_health:
            self._draw_health_bar(screen)
    
    def _draw_tank_body(self, screen: pygame.Surface, body_rect: pygame.Rect, 
                        main_color: Tuple, dark_color: Tuple):
        """Отрисовка корпуса танка"""
        cx, cy = self.x, self.y
        size = self.size
        
        # Гусеницы
        track_width = 8
        track_length = size - 4
        
        if self.direction in (Direction.UP, Direction.DOWN):
            # Левая гусеница
            pygame.draw.rect(screen, dark_color, 
                           (cx - size//2 - 2, cy - track_length//2, track_width, track_length))
            # Правая гусеница  
            pygame.draw.rect(screen, dark_color,
                           (cx + size//2 - track_width + 2, cy - track_length//2, track_width, track_length))
        else:
            # Верхняя гусеница
            pygame.draw.rect(screen, dark_color,
                           (cx - track_length//2, cy - size//2 - 2, track_length, track_width))
            # Нижняя гусеница
            pygame.draw.rect(screen, dark_color,
                           (cx - track_length//2, cy + size//2 - track_width + 2, track_length, track_width))
        
        # Корпус
        body_size = size - 12
        pygame.draw.rect(screen, main_color,
                        (cx - body_size//2, cy - body_size//2, body_size, body_size))
        
        # Башня
        tower_size = size // 2
        pygame.draw.circle(screen, dark_color, (int(cx), int(cy)), tower_size // 2 + 2)
        pygame.draw.circle(screen, main_color, (int(cx), int(cy)), tower_size // 2)
        
        # Дуло
        barrel_length = size // 2 + 5
        barrel_width = 8
        
        if self.direction == Direction.UP:
            pygame.draw.rect(screen, dark_color,
                           (cx - barrel_width//2, cy - barrel_length, barrel_width, barrel_length))
        elif self.direction == Direction.DOWN:
            pygame.draw.rect(screen, dark_color,
                           (cx - barrel_width//2, cy, barrel_width, barrel_length))
        elif self.direction == Direction.LEFT:
            pygame.draw.rect(screen, dark_color,
                           (cx - barrel_length, cy - barrel_width//2, barrel_length, barrel_width))
        elif self.direction == Direction.RIGHT:
            pygame.draw.rect(screen, dark_color,
                           (cx, cy - barrel_width//2, barrel_length, barrel_width))
    
    def _draw_health_bar(self, screen: pygame.Surface):
        """Отрисовка полоски здоровья"""
        bar_width = self.size + 10
        bar_height = 6
        x = self.x - bar_width // 2
        y = self.y - self.size // 2 - 15
        
        # Фон
        pygame.draw.rect(screen, COLORS['health_bg'], (x, y, bar_width, bar_height))
        # Здоровье
        health_width = int(bar_width * self.health / self.max_health)
        pygame.draw.rect(screen, COLORS['health_bar'], (x, y, health_width, bar_height))
        # Рамка
        pygame.draw.rect(screen, COLORS['text'], (x, y, bar_width, bar_height), 1)
    
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x - self.size//2, self.y - self.size//2, 
                          self.size, self.size)


class EnemyTank(Tank):
    """Вражеский танк с ИИ"""
    
    def __init__(self, x: float, y: float):
        super().__init__(x, y, is_player=False)
        self.direction = random.choice(list(Direction))
        
    def update(self, walls: List[pygame.Rect], player_pos: Tuple[float, float]):
        super().update(walls)
        
        self.ai_timer += 1
        
        # Простой ИИ: случайное движение + стрельба в направлении игрока
        if self.ai_timer >= self.ai_move_time:
            self.ai_timer = 0
            self.ai_move_time = random.randint(30, 90)
            
            # С вероятностью 40% - двигаться к игроку
            if random.random() < 0.4:
                dx = player_pos[0] - self.x
                dy = player_pos[1] - self.y
                
                if abs(dx) > abs(dy):
                    self.direction = Direction.RIGHT if dx > 0 else Direction.LEFT
                else:
                    self.direction = Direction.DOWN if dy > 0 else Direction.UP
            else:
                self.direction = random.choice(list(Direction))
        
        # Движение
        if not self.move(self.direction, walls):
            # Если не можем двигаться - меняем направление
            self.direction = random.choice(list(Direction))
            self.ai_timer = self.ai_move_time - 10
    
    def should_shoot(self, player_pos: Tuple[float, float]) -> bool:
        """Проверка, стоит ли стрелять"""
        if self.shoot_cooldown > 0:
            return False
            
        # Стреляем если игрок примерно на линии огня
        dx = player_pos[0] - self.x
        dy = player_pos[1] - self.y
        
        threshold = 100  # Точность прицеливания
        
        if self.direction == Direction.UP and dy < 0 and abs(dx) < threshold:
            return random.random() < 0.3
        elif self.direction == Direction.DOWN and dy > 0 and abs(dx) < threshold:
            return random.random() < 0.3
        elif self.direction == Direction.LEFT and dx < 0 and abs(dy) < threshold:
            return random.random() < 0.3
        elif self.direction == Direction.RIGHT and dx > 0 and abs(dy) < threshold:
            return random.random() < 0.3
            
        return random.random() < 0.02  # Случайная стрельба


class Explosion:
    """Эффект взрыва"""
    
    def __init__(self, x: float, y: float, size: float = 30):
        self.x = x
        self.y = y
        self.max_size = size
        self.size = 5
        self.lifetime = 20
        self.timer = 0
        self.alive = True
        self.particles = []
        
        # Частицы взрыва
        for _ in range(8):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': random.randint(3, 8),
                'color': random.choice([COLORS['explosion'], (255, 100, 50), (255, 220, 100)])
            })
    
    def update(self):
        self.timer += 1
        progress = self.timer / self.lifetime
        
        if progress < 0.3:
            self.size = self.max_size * (progress / 0.3)
        else:
            self.size = self.max_size * (1 - (progress - 0.3) / 0.7)
        
        # Обновление частиц
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2  # Гравитация
            p['size'] = max(1, p['size'] - 0.2)
        
        if self.timer >= self.lifetime:
            self.alive = False
    
    def draw(self, screen: pygame.Surface):
        if self.size > 0:
            # Основной круг взрыва
            alpha = int(200 * (1 - self.timer / self.lifetime))
            
            # Внешнее свечение
            for i in range(3):
                glow_size = self.size + i * 10
                glow_surf = pygame.Surface((int(glow_size * 2), int(glow_size * 2)), pygame.SRCALPHA)
                glow_alpha = max(0, alpha - i * 40)
                pygame.draw.circle(glow_surf, (255, 200, 100, glow_alpha), 
                                 (int(glow_size), int(glow_size)), int(glow_size))
                screen.blit(glow_surf, (int(self.x - glow_size), int(self.y - glow_size)))
            
            # Центр взрыва
            pygame.draw.circle(screen, COLORS['explosion'], 
                             (int(self.x), int(self.y)), int(self.size))
            pygame.draw.circle(screen, (255, 255, 200), 
                             (int(self.x), int(self.y)), int(self.size * 0.5))
        
        # Частицы
        for p in self.particles:
            if p['size'] > 0:
                pygame.draw.circle(screen, p['color'], 
                                 (int(p['x']), int(p['y'])), int(p['size']))


class Wall:
    """Стена/препятствие"""
    
    def __init__(self, x: int, y: int, wall_type: str = 'brick'):
        self.x = x
        self.y = y
        self.wall_type = wall_type
        self.rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
        self.health = 2 if wall_type == 'brick' else 999  # Кирпич можно разрушить
        self.alive = True
        
    def take_damage(self) -> bool:
        if self.wall_type == 'steel':
            return False  # Сталь неразрушима
        self.health -= 1
        if self.health <= 0:
            self.alive = False
            return True
        return False
    
    def draw(self, screen: pygame.Surface):
        if self.wall_type == 'brick':
            self._draw_brick(screen)
        elif self.wall_type == 'steel':
            self._draw_steel(screen)
        elif self.wall_type == 'water':
            self._draw_water(screen)
        elif self.wall_type == 'grass':
            self._draw_grass(screen)
    
    def _draw_brick(self, screen: pygame.Surface):
        # Кирпичная стена
        pygame.draw.rect(screen, COLORS['wall'], self.rect)
        
        # Узор кирпичей
        brick_h = TILE_SIZE // 3
        brick_w = TILE_SIZE // 2
        
        for row in range(3):
            offset = brick_w // 2 if row % 2 else 0
            for col in range(-1, 3):
                bx = self.x + col * brick_w + offset
                by = self.y + row * brick_h
                
                if bx >= self.x and bx + brick_w <= self.x + TILE_SIZE:
                    pygame.draw.rect(screen, COLORS['wall_dark'],
                                   (bx, by, brick_w - 1, brick_h - 1), 1)
        
        # Повреждения
        if self.health == 1:
            pygame.draw.line(screen, COLORS['wall_dark'],
                           (self.x + 5, self.y + 5),
                           (self.x + TILE_SIZE - 5, self.y + TILE_SIZE - 5), 2)
    
    def _draw_steel(self, screen: pygame.Surface):
        # Металлическая стена
        pygame.draw.rect(screen, COLORS['steel'], self.rect)
        
        # Блики
        pygame.draw.rect(screen, COLORS['steel_light'],
                        (self.x + 2, self.y + 2, TILE_SIZE - 4, 4))
        pygame.draw.rect(screen, COLORS['steel_light'],
                        (self.x + 2, self.y + 2, 4, TILE_SIZE - 4))
        
        # Болты
        bolt_size = 4
        for bx, by in [(8, 8), (TILE_SIZE - 12, 8), 
                       (8, TILE_SIZE - 12), (TILE_SIZE - 12, TILE_SIZE - 12)]:
            pygame.draw.circle(screen, (100, 100, 110),
                             (self.x + bx + bolt_size//2, self.y + by + bolt_size//2), bolt_size)
    
    def _draw_water(self, screen: pygame.Surface):
        # Вода (анимация)
        pygame.draw.rect(screen, COLORS['water'], self.rect)
        
        # Волны
        time = pygame.time.get_ticks() / 200
        for i in range(4):
            wave_y = self.y + i * 12 + 6
            offset = math.sin(time + i * 0.5) * 3
            pygame.draw.line(screen, COLORS['water_light'],
                           (self.x + 4, wave_y + offset),
                           (self.x + TILE_SIZE - 4, wave_y - offset), 2)
    
    def _draw_grass(self, screen: pygame.Surface):
        # Трава (рисуется поверх танков)
        pygame.draw.rect(screen, COLORS['grass'], self.rect)
        
        # Травинки
        for i in range(6):
            gx = self.x + 4 + i * 8
            pygame.draw.line(screen, (70, 120, 70),
                           (gx, self.y + TILE_SIZE),
                           (gx + random.randint(-2, 2), self.y + 8), 2)


# =============================================================================
# УРОВНИ
# =============================================================================

def create_level(level_num: int) -> Tuple[List[Wall], List[Tuple[int, int]]]:
    """Создание уровня. Возвращает список стен и точки спавна врагов."""
    
    walls = []
    enemy_spawns = [
        (TILE_SIZE * 2, TILE_SIZE * 2),
        (SCREEN_WIDTH // 2, TILE_SIZE * 2),
        (SCREEN_WIDTH - TILE_SIZE * 3, TILE_SIZE * 2),
    ]
    
    # Базовая структура уровня
    level_data = [
        "                     ",
        " S   B   B   B   S   ",
        "     B   B   B       ",
        " B   BBBBBBBBB   B   ",
        " B       B       B   ",
        "     S   B   S       ",
        " B       B       B   ",
        " B   BBBBBBBBB   B   ",
        "     B   B   B       ",
        " S   B   B   B   S   ",
        "                     ",
        "     B   B   B       ",
        "         B           ",
        "     BBBBBBB         ",
        "         P           ",
    ]
    
    # Добавляем дополнительные элементы в зависимости от уровня
    tile_w = SCREEN_WIDTH // 21
    tile_h = SCREEN_HEIGHT // 16
    
    for row, line in enumerate(level_data):
        for col, char in enumerate(line):
            x = col * tile_w
            y = row * tile_h
            
            if char == 'B':
                walls.append(Wall(x, y, 'brick'))
            elif char == 'S':
                walls.append(Wall(x, y, 'steel'))
            elif char == 'W':
                walls.append(Wall(x, y, 'water'))
    
    # Добавляем случайные стены в зависимости от уровня
    for _ in range(level_num * 3):
        rx = random.randint(2, 18) * tile_w
        ry = random.randint(3, 11) * tile_h
        if random.random() < 0.7:
            walls.append(Wall(rx, ry, 'brick'))
        else:
            walls.append(Wall(rx, ry, 'steel'))
    
    return walls, enemy_spawns


# =============================================================================
# ИГРА
# =============================================================================

class Game:
    """Основной класс игры"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🎮 ТАНКИ - Battle City Clone")
        self.clock = pygame.time.Clock()
        
        # Шрифты
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.reset_game()
        
    def reset_game(self):
        """Сброс игры"""
        self.level = 1
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.paused = False
        self.victory = False
        
        self._load_level()
    
    def _load_level(self):
        """Загрузка уровня"""
        self.walls, self.enemy_spawns = create_level(self.level)
        
        # Игрок в центре снизу
        self.player = Tank(SCREEN_WIDTH // 2, SCREEN_HEIGHT - TILE_SIZE * 2, is_player=True)
        
        # Враги
        self.enemies: List[EnemyTank] = []
        self.max_enemies = 3 + self.level
        self.enemies_spawned = 0
        self.enemies_to_spawn = 5 + self.level * 2
        self.spawn_timer = 0
        self.spawn_delay = 180  # 3 секунды
        
        # Снаряды и эффекты
        self.bullets: List[Bullet] = []
        self.explosions: List[Explosion] = []
    
    def spawn_enemy(self):
        """Спавн врага"""
        if len(self.enemies) < self.max_enemies and self.enemies_spawned < self.enemies_to_spawn:
            spawn_pos = random.choice(self.enemy_spawns)
            
            # Проверяем, свободна ли точка спавна
            spawn_rect = pygame.Rect(spawn_pos[0] - 20, spawn_pos[1] - 20, 40, 40)
            can_spawn = True
            
            for enemy in self.enemies:
                if enemy.get_rect().colliderect(spawn_rect):
                    can_spawn = False
                    break
            
            if self.player.get_rect().colliderect(spawn_rect):
                can_spawn = False
            
            if can_spawn:
                enemy = EnemyTank(spawn_pos[0], spawn_pos[1])
                self.enemies.append(enemy)
                self.enemies_spawned += 1
    
    def handle_events(self):
        """Обработка событий"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.game_over:
                        return False
                    self.paused = not self.paused
                
                if event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                    
        return True
    
    def update(self):
        """Обновление игры"""
        if self.game_over or self.paused:
            return
        
        # Управление игроком
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player.move(Direction.UP, [w.rect for w in self.walls if w.wall_type != 'grass'])
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player.move(Direction.DOWN, [w.rect for w in self.walls if w.wall_type != 'grass'])
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.player.move(Direction.LEFT, [w.rect for w in self.walls if w.wall_type != 'grass'])
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.player.move(Direction.RIGHT, [w.rect for w in self.walls if w.wall_type != 'grass'])
        
        if keys[pygame.K_SPACE]:
            bullet = self.player.shoot()
            if bullet:
                self.bullets.append(bullet)
        
        self.player.update([w.rect for w in self.walls if w.wall_type != 'grass'])
        
        # Спавн врагов
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_delay:
            self.spawn_timer = 0
            self.spawn_enemy()
        
        # Обновление врагов
        solid_walls = [w.rect for w in self.walls if w.wall_type not in ('grass', 'water')]
        for enemy in self.enemies:
            enemy.update(solid_walls, (self.player.x, self.player.y))
            
            if enemy.should_shoot((self.player.x, self.player.y)):
                bullet = enemy.shoot()
                if bullet:
                    self.bullets.append(bullet)
        
        # Обновление снарядов
        for bullet in self.bullets[:]:
            bullet.update()
            
            if not bullet.alive:
                self.bullets.remove(bullet)
                continue
            
            bullet_rect = bullet.get_rect()
            
            # Столкновение с стенами
            for wall in self.walls[:]:
                if wall.wall_type in ('water', 'grass'):
                    continue
                if bullet_rect.colliderect(wall.rect):
                    bullet.alive = False
                    if wall.take_damage():
                        self.walls.remove(wall)
                        self.explosions.append(Explosion(wall.x + TILE_SIZE//2, 
                                                        wall.y + TILE_SIZE//2, 20))
                    else:
                        self.explosions.append(Explosion(bullet.x, bullet.y, 15))
                    break
            
            if not bullet.alive:
                if bullet in self.bullets:
                    self.bullets.remove(bullet)
                continue
            
            # Столкновение снаряда игрока с врагами
            if bullet.is_player:
                for enemy in self.enemies[:]:
                    if bullet_rect.colliderect(enemy.get_rect()):
                        bullet.alive = False
                        if enemy.take_damage():
                            self.enemies.remove(enemy)
                            self.score += 100
                            self.explosions.append(Explosion(enemy.x, enemy.y, 40))
                        break
            else:
                # Столкновение снаряда врага с игроком
                if bullet_rect.colliderect(self.player.get_rect()):
                    bullet.alive = False
                    if self.player.take_damage():
                        self.lives -= 1
                        self.explosions.append(Explosion(self.player.x, self.player.y, 50))
                        
                        if self.lives <= 0:
                            self.game_over = True
                        else:
                            # Респавн игрока
                            self.player = Tank(SCREEN_WIDTH // 2, 
                                             SCREEN_HEIGHT - TILE_SIZE * 2, is_player=True)
        
        # Обновление взрывов
        for explosion in self.explosions[:]:
            explosion.update()
            if not explosion.alive:
                self.explosions.remove(explosion)
        
        # Проверка победы на уровне
        if self.enemies_spawned >= self.enemies_to_spawn and len(self.enemies) == 0:
            self.level += 1
            if self.level > 5:
                self.victory = True
                self.game_over = True
            else:
                self._load_level()
    
    def draw(self):
        """Отрисовка"""
        # Фон
        self.screen.fill(COLORS['bg'])
        
        # Сетка
        for x in range(0, SCREEN_WIDTH, TILE_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], (0, y), (SCREEN_WIDTH, y))
        
        # Стены (нижний слой)
        for wall in self.walls:
            if wall.wall_type != 'grass':
                wall.draw(self.screen)
        
        # Танки
        for enemy in self.enemies:
            enemy.draw(self.screen)
        
        if self.player.alive:
            self.player.draw(self.screen)
        
        # Трава (поверх танков)
        for wall in self.walls:
            if wall.wall_type == 'grass':
                wall.draw(self.screen)
        
        # Снаряды
        for bullet in self.bullets:
            bullet.draw(self.screen)
        
        # Взрывы
        for explosion in self.explosions:
            explosion.draw(self.screen)
        
        # HUD
        self._draw_hud()
        
        # Пауза
        if self.paused:
            self._draw_pause()
        
        # Game Over
        if self.game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_hud(self):
        """Отрисовка интерфейса"""
        # Фон HUD
        hud_rect = pygame.Rect(10, 10, 300, 80)
        pygame.draw.rect(self.screen, (20, 25, 20, 180), hud_rect)
        pygame.draw.rect(self.screen, COLORS['text'], hud_rect, 2)
        
        # Очки
        score_text = self.font_small.render(f"ОЧКИ: {self.score}", True, COLORS['text'])
        self.screen.blit(score_text, (20, 20))
        
        # Жизни
        lives_text = self.font_small.render(f"ЖИЗНИ: {self.lives}", True, COLORS['text'])
        self.screen.blit(lives_text, (20, 50))
        
        # Уровень
        level_text = self.font_small.render(f"УРОВЕНЬ: {self.level}", True, COLORS['text'])
        self.screen.blit(level_text, (170, 20))
        
        # Враги
        enemies_left = self.enemies_to_spawn - self.enemies_spawned + len(self.enemies)
        enemies_text = self.font_small.render(f"ВРАГОВ: {enemies_left}", True, COLORS['text'])
        self.screen.blit(enemies_text, (170, 50))
    
    def _draw_pause(self):
        """Экран паузы"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_large.render("ПАУЗА", True, COLORS['text'])
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(text, rect)
        
        hint = self.font_small.render("Нажмите ESC для продолжения", True, COLORS['text'])
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
        self.screen.blit(hint, hint_rect)
    
    def _draw_game_over(self):
        """Экран окончания игры"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if self.victory:
            title = "ПОБЕДА!"
            color = (100, 255, 100)
        else:
            title = "ИГРА ОКОНЧЕНА"
            color = (255, 100, 100)
        
        # Тень
        shadow = self.font_large.render(title, True, COLORS['text_shadow'])
        shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH // 2 + 3, SCREEN_HEIGHT // 2 - 47))
        self.screen.blit(shadow, shadow_rect)
        
        # Текст
        text = self.font_large.render(title, True, color)
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(text, rect)
        
        # Счёт
        score_text = self.font_medium.render(f"Финальный счёт: {self.score}", True, COLORS['text'])
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        self.screen.blit(score_text, score_rect)
        
        # Подсказка
        hint = self.font_small.render("R - Рестарт  |  ESC - Выход", True, COLORS['text'])
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        self.screen.blit(hint, hint_rect)
    
    def run(self):
        """Главный игровой цикл"""
        running = True
        
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🎮 ТАНКИ - Классическая аркадная игра")
    print("=" * 50)
    print("\nУправление:")
    print("  WASD / Стрелки - движение танка")
    print("  ПРОБЕЛ - стрельба")
    print("  ESC - пауза / выход")
    print("  R - рестарт после проигрыша")
    print("\nЦель: Уничтожить всех вражеских танков!")
    print("=" * 50)
    
    game = Game()
    game.run()
