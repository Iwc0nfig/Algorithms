# This is a simple implementation of Huffman encoding for text compression.

from collections import Counter
import heapq

# Function to compute the probability of each character in a given string
def get_prop(text):
    # Handle the case where the input text is empty
    if not text:
        return {}

    # Count occurrences of each character
    counts = Counter(text)
    total_length = len(text)

    # Compute probabilities by dividing occurrences by the total length
    probabilities = {char: count / total_length for char, count in counts.items()}
    return probabilities

# Define a Node class to represent each node in the Huffman Tree
class Node:
    def __init__(self, symbol, prop):
        self.symbol = symbol  # Character or None for intermediate nodes
        self.prop = prop  # Probability or weight of the node
        self.right = None  # Right child
        self.left = None   # Left child

    # Define comparison operator to allow heapq to sort nodes
    def __lt__(self, other):
        return self.prop < other.prop
    
# Function to recursively encode the characters using Huffman coding
def encoding(node, current_encode, Codetable):
    if node is None:
        return
    
    # If the node is a leaf node, store its code
    if node.symbol is not None:
        Codetable[node.symbol] = current_encode
    
    # Recursively traverse left (0) and right (1)
    encoding(node.left, current_encode + "0", Codetable)
    encoding(node.right, current_encode + "1", Codetable)

# Function to construct the Huffman Tree and generate the code table
def CreateTable(data):
    heap = []  # Min-heap to store nodes

    # Create a leaf node for each character and push it to the heap
    for symbol, prop in data.items():
        node = Node(symbol, prop)
        heapq.heappush(heap, node)
    
    # Construct the Huffman Tree by merging nodes until one remains
    while len(heap) > 1:
        leftNode = heapq.heappop(heap)  # Remove the two nodes with the lowest frequency
        rightNode = heapq.heappop(heap)
        
        mergedProp = leftNode.prop + rightNode.prop  # Combine their probabilities
        mergedNode = Node(None, mergedProp)  # Create a new intermediate node
        
        # Assign left and right children
        mergedNode.left = leftNode
        mergedNode.right = rightNode
        
        heapq.heappush(heap, mergedNode)  # Push the new node back into the heap
    
    # The last remaining node is the root of the Huffman Tree
    root = heapq.heappop(heap)
    Codetable = {}
    
    # Generate Huffman codes
    encoding(root, "", Codetable)
    return Codetable

# Function to decode an encoded string using the Huffman code table
def decode(string, code):
    # Create a reverse mapping from codes to characters
    inv_map = {v: k for k, v in code.items()}

    text = ""  # Decoded text
    current_code = ""  # Temporary storage for code bits
    
    # Traverse the encoded bit string
    for bit in string:
        current_code += bit
        
        # Check if the current sequence matches any character
        if current_code in inv_map:
            text += inv_map[current_code]
            current_code = ""  # Reset the current code
    
    return text

# Main function to demonstrate Huffman encoding and decoding
if __name__ == "__main__":
    text = "hello ello olh loh leloho"  # Input text
    prop = get_prop(text)  # Compute character probabilities
    code_map = CreateTable(prop)  # Generate Huffman codes
    enc_text = "".join(code_map[i] for i in text)  # Encode the text
    decoded = decode(enc_text, code_map)  # Decode the encoded text

    # Display the results
    print("Original Text: ", text)
    print("Frequency Map:", prop)
    print("Huffman Codes:", code_map)
    print("Encoded Text: ", enc_text)
    print("Decoded Text: ", decoded)
