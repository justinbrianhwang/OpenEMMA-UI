"""Left-side information panel for the CARLA UI."""

import pygame


class InfoPanel:
    """Renders simulation statistics and LLM I/O on the left side of the window."""

    PANEL_WIDTH = 340
    BG_COLOR = (20, 20, 30)
    BG_ALPHA = 200
    TEXT_COLOR = (255, 255, 255)
    LABEL_COLOR = (170, 170, 190)
    VALUE_COLOR = (255, 255, 255)
    BRAKE_COLOR = (255, 70, 70)
    ACCEL_COLOR = (70, 255, 100)
    IDLE_COLOR = (180, 180, 180)
    SEPARATOR_COLOR = (60, 60, 80)
    LLM_INPUT_COLOR = (100, 200, 255)   # light blue for LLM input
    LLM_OUTPUT_COLOR = (255, 200, 100)  # orange for LLM output
    NOTICE_RED_COLOR = (255, 80, 80)    # red for red light notice
    NOTICE_GREEN_COLOR = (80, 255, 80)  # green for green light notice
    NOTICE_YELLOW_COLOR = (255, 255, 80)  # yellow for yellow light notice
    NOTICE_DEFAULT_COLOR = (200, 180, 255)  # purple for other notices
    FONT_SIZE = 16
    SMALL_FONT_SIZE = 14
    TITLE_FONT_SIZE = 22
    LINE_HEIGHT = 26
    PADDING = 14

    def __init__(self, height, model_name=''):
        pygame.font.init()
        self._height = height
        self._model_name = model_name

        try:
            self._font = pygame.font.SysFont('consolas', self.FONT_SIZE)
            self._small_font = pygame.font.SysFont('consolas', self.SMALL_FONT_SIZE)
            self._title_font = pygame.font.SysFont('consolas', self.TITLE_FONT_SIZE, bold=True)
        except Exception:
            self._font = pygame.font.SysFont('monospace', self.FONT_SIZE)
            self._small_font = pygame.font.SysFont('monospace', self.SMALL_FONT_SIZE)
            self._title_font = pygame.font.SysFont('monospace', self.TITLE_FONT_SIZE, bold=True)

        # Semi-transparent background
        self._bg = pygame.Surface((self.PANEL_WIDTH, height))
        self._bg.fill(self.BG_COLOR)
        self._bg.set_alpha(self.BG_ALPHA)

    def _draw_separator(self, display, x, y):
        pygame.draw.line(display, self.SEPARATOR_COLOR,
                         (x, y), (self.PANEL_WIDTH - x, y), 1)
        return y + 10

    def _draw_label_value(self, display, x, y, label_text, value_text,
                          label_color=None, value_color=None):
        if label_color is None:
            label_color = self.LABEL_COLOR
        if value_color is None:
            value_color = self.VALUE_COLOR
        label = self._font.render(label_text, True, label_color)
        value = self._font.render(value_text, True, value_color)
        display.blit(label, (x, y))
        display.blit(value, (self.PANEL_WIDTH - self.PADDING - value.get_width(), y))
        return y + self.LINE_HEIGHT

    def _wrap_text(self, text, font, max_width):
        """Word-wrap text to fit within max_width pixels."""
        words = text.split(' ')
        lines = []
        current = ''
        for word in words:
            test = (current + ' ' + word).strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines if lines else ['']

    def render(self, display, vehicle_count, pedestrian_count, speed_kmh,
               brake_status, llm_info=None):
        """Draw the info panel onto the display surface at (0, 0).

        Args:
            display: Main pygame display surface
            vehicle_count: Number of vehicles in server
            pedestrian_count: Number of pedestrians
            speed_kmh: Current ego vehicle speed in km/h
            brake_status: 'BRAKING', 'ACCELERATING', or 'IDLE'
            llm_info: Optional dict with LLM data:
                - instruction: current instruction text (LLM input)
                - notice: current notice text
                - waypoints: 5x2 numpy array (LLM output)
                - desired_speed: float
                - curvature: float
                - steer: float
        """
        # Background
        display.blit(self._bg, (0, 0))

        x = self.PADDING
        y = self.PADDING
        max_text_w = self.PANEL_WIDTH - 2 * self.PADDING

        # Title
        if self._model_name:
            title_surf = self._title_font.render(self._model_name, True, self.TEXT_COLOR)
            display.blit(title_surf, (x, y))
            y += self.LINE_HEIGHT + 8

        y = self._draw_separator(display, x, y)

        # ── World Info ──
        section_surf = self._font.render('[ World Info ]', True, self.LABEL_COLOR)
        display.blit(section_surf, (x, y))
        y += self.LINE_HEIGHT

        y = self._draw_label_value(display, x, y, 'Vehicles:', str(vehicle_count))
        y = self._draw_label_value(display, x, y, 'Pedestrians:', str(pedestrian_count))
        y += 4

        y = self._draw_separator(display, x, y)

        # ── Vehicle Status ──
        section_surf = self._font.render('[ Vehicle Status ]', True, self.LABEL_COLOR)
        display.blit(section_surf, (x, y))
        y += self.LINE_HEIGHT

        y = self._draw_label_value(display, x, y, 'Speed:', f'{speed_kmh:.1f} km/h')

        # Brake/Accel status
        if brake_status == 'BRAKING':
            status_color = self.BRAKE_COLOR
        elif brake_status == 'ACCELERATING':
            status_color = self.ACCEL_COLOR
        else:
            status_color = self.IDLE_COLOR
        y = self._draw_label_value(display, x, y, 'Status:', brake_status,
                                   value_color=status_color)

        if llm_info:
            # Steer & desired speed from model
            steer = llm_info.get('steer', 0.0)
            desired_spd = llm_info.get('desired_speed', 0.0)
            curvature = llm_info.get('curvature', 0.0)
            y = self._draw_label_value(display, x, y, 'Steer:', f'{steer:+.3f}')
            y = self._draw_label_value(display, x, y, 'Des.Speed:',
                                       f'{desired_spd:.1f} m/s')
            y = self._draw_label_value(display, x, y, 'Curvature:', f'{curvature:.3f}')

        y += 4
        y = self._draw_separator(display, x, y)

        # ── LLM Input/Output ──
        if llm_info:
            section_surf = self._font.render('[ LLM Input ]', True, self.LLM_INPUT_COLOR)
            display.blit(section_surf, (x, y))
            y += self.LINE_HEIGHT

            # Instruction (word-wrapped)
            instruction = llm_info.get('instruction', '')
            if instruction:
                lines = self._wrap_text(instruction, self._small_font, max_text_w)
                for line in lines[:3]:  # max 3 lines
                    surf = self._small_font.render(line, True, self.TEXT_COLOR)
                    display.blit(surf, (x, y))
                    y += self.LINE_HEIGHT - 4

            # Notice (traffic light / scenario) — color-coded
            notice = llm_info.get('notice', '')
            if notice:
                y += 2
                notice_lower = notice.lower()
                if 'red' in notice_lower:
                    notice_color = self.NOTICE_RED_COLOR
                elif 'green' in notice_lower:
                    notice_color = self.NOTICE_GREEN_COLOR
                elif 'yellow' in notice_lower:
                    notice_color = self.NOTICE_YELLOW_COLOR
                else:
                    notice_color = self.NOTICE_DEFAULT_COLOR
                label = self._small_font.render('Notice:', True, notice_color)
                display.blit(label, (x, y))
                y += self.LINE_HEIGHT - 4
                notice_lines = self._wrap_text(notice, self._small_font, max_text_w)
                for line in notice_lines[:2]:
                    surf = self._small_font.render(line, True, notice_color)
                    display.blit(surf, (x, y))
                    y += self.LINE_HEIGHT - 4

            y += 6
            y = self._draw_separator(display, x, y)

            section_surf = self._font.render('[ LLM Output ]', True, self.LLM_OUTPUT_COLOR)
            display.blit(section_surf, (x, y))
            y += self.LINE_HEIGHT

            # Waypoints
            waypoints = llm_info.get('waypoints')
            if waypoints is not None:
                for i in range(min(5, len(waypoints))):
                    wp_text = f'WP{i}: ({waypoints[i][0]:+6.2f}, {waypoints[i][1]:+6.2f})'
                    surf = self._small_font.render(wp_text, True, self.LLM_OUTPUT_COLOR)
                    display.blit(surf, (x, y))
                    y += self.LINE_HEIGHT - 4
