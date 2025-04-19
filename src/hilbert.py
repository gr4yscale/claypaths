import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def distance_from_coordinates(x, y, order):
    """
    Convert coordinates to distance along Hilbert curve.
    
    Args:
        x (float): X coordinate (normalized 0-1)
        y (float): Y coordinate (normalized 0-1)
        order (int): Order of the curve
        
    Returns:
        int: Distance along curve
    """
    # Scale coordinates to integer grid
    max_val = 2 ** order
    x_int = int(x * max_val)
    y_int = int(y * max_val)
    logger.debug(f"Converting coordinates (x={x:.4f}, y={y:.4f}) to "
               f"integer grid (x_int={x_int}, y_int={y_int})")
    
    # Initialize distance and direction
    d = 0
    direction = 'd'
    
    # Walk through each bit position
    for i in range(order-1, -1, -1):
        # Get current quadrant
        x_bit = (x_int >> i) & 1
        y_bit = (y_int >> i) & 1
        bits = (x_bit << 1) | y_bit
        
        # Update distance based on current direction
        if direction == 'd':
            if bits == 0:
                d += 0
                direction = 'r'
            elif bits == 1:
                d += 1 << (2*i)
            elif bits == 2:
                d += 3 << (2*i)
            else:  # bits == 3
                d += 2 << (2*i)
                direction = 'l'
        elif direction == 'r':
            if bits == 0:
                d += 0
                direction = 'd'
            elif bits == 1:
                d += 1 << (2*i)
            elif bits == 2:
                d += 3 << (2*i)
            else:  # bits == 3
                d += 2 << (2*i)
                direction = 'u'
        elif direction == 'l':
            if bits == 0:
                d += 3 << (2*i)
                direction = 'd'
            elif bits == 1:
                d += 2 << (2*i)
            elif bits == 2:
                d += 0 << (2*i)
            else:  # bits == 3
                d += 1 << (2*i)
                direction = 'u'
        else:  # direction == 'u'
            if bits == 0:
                d += 3 << (2*i)
                direction = 'l'
            elif bits == 1:
                d += 2 << (2*i)
            elif bits == 2:
                d += 0 << (2*i)
            else:  # bits == 3
                d += 1 << (2*i)
                direction = 'r'
    
    return d

def coordinates_from_distance(d, order):
    """
    Convert distance along Hilbert curve to coordinates.
    
    Args:
        d (int): Distance along curve
        order (int): Order of the curve
        
    Returns:
        tuple: (x, y) coordinates (normalized 0-1)
    """
    # Convert distance to binary representation
    bits = format(d, f'0{2*order}b')
    logger.debug(f"Converting distance {d} to {2*order}-bit binary: {bits}")
    
    # Initialize coordinates and direction
    x, y = 0, 0
    direction = 'd'
    
    # Walk through each bit pair
    for i in range(0, 2*order, 2):
        bits_pair = bits[-(i+2):-(i) if i > 0 else None]
        
        # Update coordinates based on current direction
        if direction == 'd':
            if bits_pair == '00':
                x, y = y, x
                direction = 'r'
            elif bits_pair == '01':
                y += 1
            elif bits_pair == '10': 
                x += 1
                y += 1
            else:  # '11'
                x, y = 1 - y, 1 - x
                direction = 'l'
        elif direction == 'r':
            if bits_pair == '00':
                x, y = y, x
                direction = 'd'
            elif bits_pair == '01':
                x += 1
                y
                direction = 'd'
            elif bits_pair == '10':
                y += 1
                direction = 'd'
            else:  # '11'
                x, y = 1 - y, 1 - x
                direction = 'u'
        elif direction == 'l':
            if bits_pair == '00':
                x, y = 1 - y, 1 - x
                direction = 'd'
            elif bits_pair == '01':
                y += 1
                direction = 'd'
            elif bits_pair == '10':
                x += 1
                direction = 'd'
            else:  # '11':
                x, y = y, x
                direction = 'u'
        else:  # 'u'
            if bits_pair == '00':
                x, y = 1 - y, 1 - x
                direction = 'l'
            elif bits_pair == '01':
                x += 1
                direction = 'r'
            elif bits_pair == '10':
                y += 1
                direction = 'l'
            else:  # '11':
                x, y = y, x
                direction = 'd'
    
    # Normalize coordinates
    max_val = 2 ** order
    return x / max_val, y / max_val
