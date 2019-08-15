import queue

from screeninfo import screeninfo


class Monitor:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    def to_dict(self):
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }

    def copy(self):
        return Monitor(self.left, self.top, self.width, self.height)


def mon_iterator(n, width, height, top_spacing=0):
    """
    Yields n Monitors with the given width and height

    :param n: The number of Monitors to create
    :param width: The width of every Monitor
    :param height: The height of every Monitor
    :param top_spacing: The spacing over the monitors
    :return: An iterator over Monitors that do no overlap and have the given dimensions
    """
    screen_width = screeninfo.get_monitors()[0].width
    screen_height = screeninfo.get_monitors()[0].height

    x = 0
    y = top_spacing

    for i in range(n):
        yield Monitor(x, y, width, height)

        x += width

        if x + width > screen_width:
            x = 0
            y += height

        if y + height > screen_height:
            raise ToManyMonitorsError(
                'Could not create {} monitors of size {}, because insufficient screen space'.format(n, (width, height))
            )


def get_all_from_queue(q: queue.Queue) -> list:
    result_list = []
    while True:
        try:
            result_list.append(q.get_nowait())
        except queue.Empty:
            break
    return result_list


class ToManyMonitorsError(Exception):
    pass
