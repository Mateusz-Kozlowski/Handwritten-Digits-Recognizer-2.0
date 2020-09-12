import pygame as pg
import tensorflow as tf
import numpy as np
import os
import cv2 as cv

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)  # Set the window position
pg.init()

# Let's define some constants:
WHITE = (255, 255, 255)
GREY240 = (240, 240, 240)
GREY224 = (224, 224, 224)
GREY192 = (192, 192, 192)
GREY160 = (160, 160, 160)
GREY128 = (128, 128, 128)
GREY96 = (96, 96, 96)
GREY64 = (64, 64, 64)
GREY32 = (32, 32, 32)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
BACKGROUND_COLOR = GREY240

FPS = 120
LEFT_MARGIN = 64
TOP_MARGIN = 16
INPUT_TOP = 2 * TOP_MARGIN + 32 + 1
INPUT_WIDTH = 280 * 5
INPUT_HEIGHT = 280
WIDTH = 2 * LEFT_MARGIN + INPUT_WIDTH
HEIGHT = 784

window = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("HANDWRITTEN NUMBERS RECOGNITION")

model = tf.keras.models.load_model('cnn model.h5', custom_objects=None, compile=True)


class Button:
	def __init__(
			self, x_pos, y_pos, width, height,
			idle_color, hover_color, pressed_color,
			outline_thickness, text_color,
			char_size, text='', font='arial'
	):
		self.xPos = int(x_pos)
		self.yPos = int(y_pos)
		self.width = int(width)
		self.height = int(height)
		self.idleColor = idle_color
		self.hoverColor = hover_color
		self.pressedColor = pressed_color
		self.lightOutlineColor = (255, 255, 255)
		self.darkOutlineColor = (0, 0, 0)
		self.outlineThickness = outline_thickness
		self.textColor = text_color
		self.charSize = char_size
		self.text = text
		self.font = font
		self.over = False
		self.clicked = False
		self.pressed = False

	def update(self, pos):
		self.clicked = False
		if self.xPos < pos[0] < self.xPos + self.width and self.yPos < pos[1] < self.yPos + self.height:
			self.over = True
			if pg.mouse.get_pressed()[0]:
				if not self.pressed:
					self.clicked = True
				self.pressed = True
			else:
				self.pressed = False
		else:
			self.over = False
			self.pressed = False

	def get_clicked(self):
		return self.clicked

	def get_text(self):
		return self.text

	def render(self):
		# DRAWING TOP-LEFT OUTLINE AS A RECTANGLE WHICH WILL BE ALMOST COMPLETELY COVERED:
		pg.draw.rect(window, self.lightOutlineColor, (self.xPos, self.yPos, self.width, self.height))

		# DRAWING BOTTOM-RIGHT OUTLINE AS RECTANGLE WHICH COVERS THE PREVIOUS ONE:
		t = self.outlineThickness
		pg.draw.rect(window, self.darkOutlineColor, (self.xPos + t, self.yPos + t, self.width - t, self.height - t))

		# DRAWING CENTER OF THE CHECK-BOX:
		if self.pressed:
			pg.draw.rect(
				window, self.pressedColor, (self.xPos + t, self.yPos + t, self.width - 2 * t, self.height - 2 * t)
			)
		elif self.over:
			pg.draw.rect(
				window, self.hoverColor, (self.xPos + t, self.yPos + t, self.width - 2 * t, self.height - 2 * t)
			)
		else:
			pg.draw.rect(
				window, self.idleColor, (self.xPos + t, self.yPos + t, self.width - 2 * t, self.height - 2 * t)
			)

		# DRAWING TEXT:
		if self.text != '':
			font = pg.font.SysFont(self.font, self.charSize)
			text = font.render(self.text, 1, self.textColor)
			window.blit(
				text,
				(
					(self.xPos + (self.width - text.get_width()) // 2),
					(self.yPos + (self.height - text.get_height()) // 2)
				)
			)


class Digit:
	def __init__(self):
		self.xPos = None
		self.yPos = None
		self.label = None
		self.percentage = None

	def set_x_pos(self, x_pos):
		self.xPos = x_pos

	def set_y_pos(self, y_pos):
		self.yPos = y_pos


class InputField:
	def __init__(self, x_pos, y_pos, width, height):
		self.xPos = x_pos
		self.yPos = y_pos
		self.width = width
		self.height = height
		self.prePredictedCanvas = None
		self.digits = list()
		self.labelsSortedByX = list()

	def predict(self):
		self.prePredictedCanvas = cut_from_the_window(self.xPos, self.yPos, self.width, self.height)
		canvas_file_path = 'Canvas images/pre_predicted_canvas.png'
		pg.image.save(self.prePredictedCanvas, canvas_file_path)

		self.digits.clear()
		self.labelsSortedByX.clear()

		image2find_contours = cv.imread(canvas_file_path)
		image2find_contours = cv.cvtColor(image2find_contours, cv.COLOR_BGR2GRAY)
		ret, th = cv.threshold(image2find_contours, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
		contours = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
		index = -1
		for cnt in contours:
			index += 1
			x, y, w, h = cv.boundingRect(cnt)
			the_digit_image = pg.Surface((w, h))
			the_digit_image.blit(window, (0, 0), (x + self.xPos, y + self.yPos, w, h))
			pg.image.save(the_digit_image, 'Canvas images/' + str(index) + ' contour.png')
			the_digit_image = cv.imread('Canvas images/' + str(index) + ' contour.png')

			top_padding = 0
			bottom_padding = 0
			left_padding = 0
			right_padding = 0
			if h > w:
				left_padding = (h - w) // 2
				right_padding = left_padding
			else:
				top_padding = (w - h) // 2
				bottom_padding = top_padding
			a = max(w, h)
			the_digit_image = cv.copyMakeBorder(
				the_digit_image,
				top_padding, bottom_padding, left_padding, right_padding,
				cv.BORDER_CONSTANT, value=WHITE
			)
			the_digit_image = cv.copyMakeBorder(
				the_digit_image,
				a // 4, a // 4, a // 4, a // 4,
				cv.BORDER_CONSTANT, value=WHITE
			)

			pg.draw.rect(window, BLUE, (x + self.xPos, y + self.yPos, w, 1))
			pg.draw.rect(window, BLUE, (x + self.xPos, y + self.yPos + h, w, 1))
			pg.draw.rect(window, BLUE, (x + self.xPos, y + self.yPos, 1, h))
			pg.draw.rect(window, BLUE, (x + self.xPos + w, y + self.yPos, 1, h))

			the_digit_image = cv.cvtColor(the_digit_image, cv.COLOR_BGR2GRAY)
			the_digit_image = cv.bitwise_not(the_digit_image)
			the_digit_image = cv.resize(the_digit_image, (28, 28), interpolation=cv.INTER_AREA)
			the_digit_image = the_digit_image.reshape(1, 28, 28, 1)
			the_digit_image = the_digit_image / 255.0

			predictions = model.predict([the_digit_image])[0]

			self.digits.append(Digit())
			self.digits[-1].set_x_pos(x)
			self.digits[-1].set_y_pos(y)
			self.digits[-1].label = np.argmax(predictions)
			self.digits[-1].percentage = int(max(predictions) * 100)

			self.labelsSortedByX.append((x, np.argmax(predictions)))

		self.labelsSortedByX = sorted(self.labelsSortedByX)

	def unpredict(self):
		pg.draw.rect(window, WHITE, (self.xPos, self.yPos, self.width, self.height))
		if self.prePredictedCanvas is not None:
			window.blit(self.prePredictedCanvas, (self.xPos, self.yPos, self.width, self.height))

	def get_digits(self):
		return self.digits

	def get_labels_sorted_by_x(self):
		return self.labelsSortedByX


def init_buttons():
	buttons = list()
	x = (WIDTH - 2 * 128 - 32) / 2
	y = TOP_MARGIN
	buttons.append(
		Button(
			x + 0 * (128 + 32), y, 128, 32,
			GREY224, GREY240, GREY192, 1, BLACK, 18, "BRUSH"
		)
	)
	buttons.append(
		Button(
			x + 1 * (128 + 32), y, 128, 32,
			GREY224, GREY240, GREY192, 1, BLACK, 18, "RUBBER"
		)
	)
	return buttons


def endless_display():
	# BACKGROUND:
	window.fill(BACKGROUND_COLOR)

	# CANVAS:
	pg.draw.rect(
		window, WHITE,
		(
			LEFT_MARGIN, INPUT_TOP,
			INPUT_WIDTH, INPUT_HEIGHT
		)
	)


def change_tool(buttons, tool):
	if pg.mouse.get_pressed()[0]:
		for btn in buttons:
			if btn.get_clicked():
				text = btn.get_text()
				if text == 'BRUSH':
					tool = 'BRUSH'
				elif text == 'RUBBER':
					tool = 'RUBBER'
				break
	return tool


def draw_round_line(start, end, color, canvas_rect, radius=12):
	if start is None or end is None:
		return
	dx = end[0] - start[0]
	dy = end[1] - start[1]

	distance = max(abs(dx), abs(dy))
	for i in range(distance):
		x = int(start[0] + float(i) / distance * dx)
		y = int(start[1] + float(i) / distance * dy)
		xx, yy, w, h = canvas_rect
		if xx - radius <= x < xx + radius + w:
			if yy - radius <= y < yy + radius + h:
				pg.draw.circle(window, color, (x, y), radius)
	else:
		x = start[0]
		y = start[1]
		xx, yy, w, h = canvas_rect
		if xx - radius <= x < xx + radius + w:
			if yy - radius <= y < yy + radius + h:
				pg.draw.circle(window, color, (x, y), radius)


def cut_from_the_window(x, y, width, height):
	clipping = pg.Surface((width, height))
	clipping.blit(window, (0, 0), (x, y, width, height))
	return clipping


def draw_part_of_background():
	# REDRAWING SOME BACKGROUND TO COVER ANY EXCESS BRUSH AND ERASER RESIDUE:
	pg.draw.rect(window, BACKGROUND_COLOR, (0, 0, WIDTH, INPUT_TOP))
	pg.draw.rect(window, BACKGROUND_COLOR, (0, INPUT_TOP + INPUT_HEIGHT, WIDTH, HEIGHT - INPUT_TOP - INPUT_HEIGHT))
	pg.draw.rect(window, BACKGROUND_COLOR, (0, 0, LEFT_MARGIN, HEIGHT))
	pg.draw.rect(
		window,
		BACKGROUND_COLOR,
		(
			LEFT_MARGIN + INPUT_WIDTH,
			0,
			WIDTH - LEFT_MARGIN + INPUT_WIDTH,
			HEIGHT
		)
	)


def draw_powers_labels(sorted_labels, x_center, y_center, char_size):
	string = ''
	index = 0
	for sorted_label in sorted_labels:
		x, label = sorted_label
		if index == 0:
			string += str(label) + '*10^' + str(len(sorted_labels) - 1 - index)
		else:
			string += ' + ' + str(label) + '*10^' + str(len(sorted_labels) - 1 - index)
		index += 1

	font = pg.font.SysFont('arial', char_size)
	text = font.render(string, 1, BLACK)
	font = pg.font.SysFont('arial', char_size * min(WIDTH - 2 * LEFT_MARGIN, text.get_width()) // text.get_width())
	text = font.render(string, 1, BLACK)
	window.blit(text, (x_center - text.get_width() // 2, y_center - text.get_height() // 2))


def get_decimal_number(sorted_labels):
	number = 0
	index = 0
	for sorted_label in sorted_labels:
		x, label = sorted_label
		number += label * 10 ** (len(sorted_labels) - 1 - index)
		index += 1
	return number


def draw_decimal_labels(decimal_number, x_center, y_center, char_size):
	string = str(decimal_number)
	font = pg.font.SysFont('arial', char_size)
	text = font.render(string, 1, BLACK)
	font = pg.font.SysFont('arial', char_size * min(WIDTH - 2 * LEFT_MARGIN, text.get_width()) // text.get_width())
	text = font.render(string, 1, BLACK)
	window.blit(text, (int(x_center - text.get_width() // 2), int(y_center - text.get_height() // 2)))


def draw_binary_labels(decimal_number, x_center, y_center, char_size):
	binary = bin(decimal_number)
	string = str(binary)
	font = pg.font.SysFont('arial', char_size)
	text = font.render(string, 1, BLACK)
	font = pg.font.SysFont('arial', char_size * min(WIDTH - 2 * LEFT_MARGIN, text.get_width()) // text.get_width())
	text = font.render(string, 1, BLACK)
	window.blit(text, (int(x_center - text.get_width() // 2), int(y_center - text.get_height() // 2)))


def draw_hexadecimal_labels(decimal_number, x_center, y_center, char_size):
	string = ''
	while decimal_number != 0:
		rest = decimal_number % 16
		if rest >= 10:
			rest += 55
			string += chr(rest)
		else:
			string += str(rest)
		decimal_number = decimal_number // 16
	string += 'x0'
	string = string[::-1]
	font = pg.font.SysFont('arial', char_size)
	text = font.render(string, 1, BLACK)
	font = pg.font.SysFont('arial', char_size * min(WIDTH - 2 * LEFT_MARGIN, text.get_width()) // text.get_width())
	text = font.render(string, 1, BLACK)
	window.blit(text, (int(x_center - text.get_width() // 2), int(y_center - text.get_height() // 2)))


def draw_english_labels(decimal_number, x_center, y_center, char_size):
	pass


def draw(buttons, input_field, show_labels):
	draw_part_of_background()

	for btn in buttons:
		btn.render()

	# DRAWING LABELS IN CANVAS:
	if show_labels:
		font = pg.font.SysFont('Arial', 28)
		digits = input_field.get_digits()
		for i in range(int(len(digits))):
			text = font.render(str(digits[i].label) + ' ' + str(digits[i].percentage) + '%', True, BLUE)
			window.blit(
				text,
				(
					int(digits[i].xPos + LEFT_MARGIN),
					int(digits[i].yPos - 1.25 * text.get_height() + INPUT_TOP)
				)
			)

	# DRAWING LABELS UNDER THE CANVAS:
	if len(input_field.get_digits()) > 0:
		draw_powers_labels(input_field.get_labels_sorted_by_x(), WIDTH // 2, 2 * INPUT_TOP + INPUT_HEIGHT, INPUT_TOP)
		decimal_number = get_decimal_number(input_field.get_labels_sorted_by_x())
		draw_decimal_labels(decimal_number, WIDTH // 2, 3.5 * INPUT_TOP + INPUT_HEIGHT, INPUT_TOP)
		draw_binary_labels(decimal_number, WIDTH // 2, 5 * INPUT_TOP + INPUT_HEIGHT, INPUT_TOP)
		draw_hexadecimal_labels(decimal_number, WIDTH // 2, 6.5 * INPUT_TOP + INPUT_HEIGHT, INPUT_TOP)
		draw_english_labels(decimal_number, WIDTH // 2, 8 * INPUT_TOP + INPUT_HEIGHT, INPUT_TOP)

	pg.display.update()


def main():
	buttons = init_buttons()
	input_field = InputField(LEFT_MARGIN, INPUT_TOP, INPUT_WIDTH, INPUT_HEIGHT)
	endless_display()

	last_mouse_pos = None
	tool = 'BRUSH'
	drawing = False
	show_labels = True
	clock = pg.time.Clock()
	run = True

	while run:
		clock.tick(FPS)
		mouse_pos = pg.mouse.get_pos()

		for event in pg.event.get():
			if event.type == pg.QUIT:
				run = False

		for btn in buttons:
			btn.update(mouse_pos)

		if not drawing:
			tool = change_tool(buttons, tool)

		if drawing:
			if not pg.mouse.get_pressed()[0]:
				drawing = False
				show_labels = True
				input_field.predict()
			else:
				rect = (LEFT_MARGIN, INPUT_TOP, INPUT_WIDTH, INPUT_HEIGHT)
				if tool == 'BRUSH':
					draw_round_line(mouse_pos, last_mouse_pos, BLACK, rect, 8)
				elif tool == 'RUBBER':
					draw_round_line(mouse_pos, last_mouse_pos, WHITE, rect, 32)
		else:
			if pg.mouse.get_pressed()[0]:
				x, y = mouse_pos
				if LEFT_MARGIN <= x < LEFT_MARGIN + INPUT_WIDTH:
					if INPUT_TOP <= y < INPUT_TOP + INPUT_HEIGHT:
						drawing = True
						show_labels = False
						input_field.unpredict()

		last_mouse_pos = mouse_pos

		draw(buttons, input_field, show_labels)

	pg.quit()


if __name__ == '__main__':
	main()

print('Code is done, so everything works fine!')
