#knapsack.py by Kendal Fisher and Evan Hatton

import sys

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " [input digit]")
    exit(1)


def getData(idx):
  if idx < 10:
    idx = "0" + str(idx)
  f = open("KnapsackTestData/p" + str(idx) + "_v.txt", 'r')
  vals = f.readlines()
  vals = list(map(str.strip, vals))
  vals = list(map(int, vals))
  f.close()
  f = open("KnapsackTestData/p" + str(idx) + "_w.txt", 'r')
  lbs = f.readlines()
  lbs = list(map(str.strip, lbs))
  lbs = list(map(int, lbs))
  f.close()
  f = open("KnapsackTestData/p" + str(idx) + "_c.txt", 'r')
  W = f.readlines()
  W = int(W[0].strip())
  f.close()
  print("File containing the capacity, weights, and values are: p" + idx + "_c.txt, p" + idx + "_w.txt and p" + idx + "_v.txt\n")
  return (vals, lbs, W)

gd = getData(int(sys.argv[1]))
print("Knapsack capacity = " + str(gd[2]) + ". Total number of items = " + str(len(gd[0])) + "\n")

def tblSack(v, w, c):
  grid = []
  grid.append([])
  for x in range(c+1):
    grid[0].append(0)
  for x in range(1, len(v) + 1):
    grid.append([])
    for y in range(c+1):
      if (y - w[x-1]) >= 0:
        grid[x].append(
            max(grid[x-1][y], v[x-1] + grid[x-1][y - w[x-1]])
        )
      else:
        grid[x].append(grid[x-1][y])

  return grid

def knapsackGrid(grid):
  return grid[len(grid) - 1][len(grid[0]) - 1]

def tblOps(v, w, c):
  ops = 0
  for x in range(len(v)):
    for y in range(c):
      ops += 1
  return ops

tbl = tblSack(gd[0], gd[1], gd[2])

simpleset = []
def find_subset_1a(i, j):
    if tbl[i][j] != 0:
        if j < gd[1][i-1]:
            find_subset_1a(i - 1, j)
        elif tbl[i][j] > tbl[i - 1][j]:
            simpleset.insert(0, i)
            find_subset_1a(i - 1, j - gd[1][i-1])
        else:
            find_subset_1a(i - 1, j)

find_subset_1a(len(tbl) - 1, gd[2])

print("(1a) Traditional Dynamic Programming Optimal value: " + str(knapsackGrid(tbl)))
print("(1a) Traditional Dynamic Programming Optimal subset: " + str(simpleset))
print("(1a) Traditional Dynamic Programming Total Basic Ops: " + str(tblOps(gd[0], gd[1], gd[2])) + "\n")

def ratio(v, w):
  r = []
  for x in range(len(v)):
    y = v[x]/w[x]
    r.append((y, x))
  return r

#quicksort implementation edited from https://www.geeksforgeeks.org/python-program-for-quicksort/
def partition(l, r, nums):
    pivot, ptr = nums[r], l
    for i in range(l, r):
        if nums[i][0] >= pivot[0]:
            nums[i], nums[ptr] = nums[ptr], nums[i]
            ptr += 1
    nums[ptr], nums[r] = nums[r], nums[ptr]
    return ptr

def quicksort(l, r, nums):
    if len(nums) == 1:
        return nums
    if l < r:
        pi = partition(l, r, nums)
        quicksort(l, pi-1, nums)
        quicksort(pi+1, r, nums)
    return nums

def greedySack(v, w, c):
  r = quicksort(0, len(w) - 1, ratio(v, w))
  curW = c
  ptr = -1
  for x in range(len(v)):
    if curW - w[r[x][1]] >= 0:
      curW = curW - w[r[x][1]]
    else:
      ptr = x - 1
      break
  idx = list(map(lambda h: h[1], r[:ptr + 1]))
  vals = list(map(lambda h: v[h[1]], r[:ptr + 1]))
  return sum(vals), idx

QSops = 0
def partOps(l, r, nums):
    global QSops
    pivot, ptr = nums[r], l
    for i in range(l, r):
        QSops += 1
        if nums[i][0] >= pivot[0]:
            nums[i], nums[ptr] = nums[ptr], nums[i]
            ptr += 1
    nums[ptr], nums[r] = nums[r], nums[ptr]
    return ptr

def quickOps(l, r, nums):
    global QSops
    if len(nums) == 1:
        return nums, 0
    if l < r:
        QSops += 1
        pi = partOps(l, r, nums)
        quickOps(l, pi-1, nums)
        quickOps(pi+1, r, nums)
    return nums

def greedySackOps(v, w, c):
  global QSops
  QSops = 0
  r = quickOps(0, len(w) - 1, ratio(v, w))
  curW = c
  ptr = -1
  for x in range(len(v)):
    QSops += 1
    if curW - w[r[x][1]] >= 0:
      curW = curW - w[r[x][1]]
    else:
      ptr = x - 1
      break
  return QSops


import heapq
# from standard library: https://github.com/python/cpython/blob/3.10/Lib/heapq.py
keys = {}
def ratios(v, w):
  global keys
  r = []
  for x in range(len(v)):
    y = v[x]/w[x]
    r.append(y)
    keys[y] = x
  return r

def heapSack(v, w, c):
  global keys
  r = ratios(v, w)
  heapq._heapify_max(r)
  curW = c
  ptr = -1
  vals = []
  for x in range(len(v)):
    itm = heapq._heappop_max(r)
    idx = keys[itm]
    if curW - w[idx] >= 0:
      curW = curW - w[idx]
      vals.append(itm)
    else:
      break
  idxs = list(map(lambda h: keys[h], vals))
  vals = list(map(lambda h: v[h], idxs))
  return sum(vals), idxs

# modified from standard library source https://github.com/python/cpython/blob/3.10/Lib/heapq.py
# in order to count operations

HSops = 0
def _heapify_max_ops(x):
    global HSops
    """Transform list into a maxheap, in-place, in O(len(x)) time."""
    n = len(x)
    for i in reversed(range(n//2)):
        HSops += 1
        _siftup_max_ops(x, i)
    return HSops

# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
def _siftdown_ops(heap, startpos, pos):
    global HSops
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        HSops += 1
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _siftup_max_ops(heap, pos):
    global HSops
    'Maxheap variant of _siftup'
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the larger child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of larger child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos
        # Move the larger child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
        HSops ++ 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown_max_ops(heap, startpos, pos)

def _siftdown_max_ops(heap, startpos, pos):
    global HSops
    'Maxheap variant of _siftdown'
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        HSops += 1
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem

def _heappop_max_ops(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup_max_ops(heap, 0)
        return returnitem
    return lastelt

def heapSackOps(v, w, c):
  global keys
  global HSops
  HSops = 0
  r = ratios(v, w)
  _heapify_max_ops(r)
  curW = c
  ptr = -1
  vals = []
  for x in range(len(v)):
    itm = _heappop_max_ops(r)
    idx = keys[itm]
    HSops += 1
    if curW - w[idx] >= 0:
      curW = curW - w[idx]
      vals.append(itm)
    else:
      break
  return HSops

values = gd[0].copy()
weights = gd[1].copy()
capacity = gd[2]
values.insert(0, 0)
weights.insert(0, 0)


def init_table(v, W):
    table = []
    for row in range(0, len(v)):
        temp = []
        for col in range(0, W + 1):
            if col == 0 or row == 0:
                temp.append(0)
            else:
                temp.append(-1)
        table.append(temp)
    return table


table = init_table(values, capacity)

count_mem = [0]


def memory_knapsack(i, j):
    count_mem[0] += 1
    if table[i][j] < 0:
        count_mem[0] += 1
        if j < weights[i]:
            temp = memory_knapsack(i - 1, j)
        else:
            temp = max(memory_knapsack(i - 1, j), values[i] + memory_knapsack(i - 1, j - weights[i]))
        table[i][j] = temp
    return table[i][j]


result = memory_knapsack(len(values) - 1, capacity)
print("(1b) Memory-function Dynamic Programming Optimal value:", result)

subset = []


def find_subset(i, j):
    count_mem[0] += 1
    if table[i][j] != 0:
        if j < weights[i]:
            count_mem[0] += 1
            find_subset(i - 1, j)
        elif table[i][j] > table[i - 1][j]:
            count_mem[0] += 1
            subset.insert(0, i)
            find_subset(i - 1, j - weights[i])
        else:
            count_mem[0] += 1
            find_subset(i - 1, j)


find_subset(len(values) - 1, capacity)
print("(1b) Memory-function Dynamic Programming Optimal subset:", '{' + ', '.join(map(str, subset)) + '}')
print("(1b) Memory-function Dynamic Programming Total Basic Ops:" + str(count_mem[0]) + "\n")

k = round(len(values) * (capacity+1) * .5)
hashes = []
count_hash = [0]
count_nodes = [0]


class Node:

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None

    def find_index(self, i, j):
        temp = self.head
        while temp:
            count_hash[0] += 1
            if temp.data[0] == i and temp.data[1] == j:
                return temp.data[2]
            temp = temp.next
        return -1

    def insert_front(self, new_data):
        count_nodes[0] += 1
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node


def hash_func(i, j):
    return ((i-1) * capacity + j) % k


def init_hashes():
    for x in range(0, k):
        temp = LinkedList()
        hashes.append(temp)


def hash_knapsack(i, j):
    if i == 0 or j == 0:
        count_hash[0] += 2
        return 0
    pos = hash_func(i, j)
    value = hashes[pos].find_index(i, j)
    count_hash[0] += 1
    if value == -1:
        count_hash[0] += 1
        if j < weights[i]:
            temp = hash_knapsack(i - 1, j)
        else:
            temp = max(hash_knapsack(i - 1, j), values[i] + hash_knapsack(i - 1, j - weights[i]))
        hashes[pos].insert_front((i, j, temp))
        return temp
    return value


init_hashes()
result = hash_knapsack(len(values) - 1, capacity)
print("(1c) Space-efficient Dynamic Programming Optimal value:", result)

hash_subset = []


def find_hash_subset(i, j):
    pos = hash_func(i, j)
    value = hashes[pos].find_index(i, j)
    count_hash[0] += 1
    if value > 0:
        pos_above = hash_func(i - 1, j)
        value_above = hashes[pos_above].find_index(i-1, j)
        if j < weights[i]:
            count_hash[0] += 1
            find_hash_subset(i - 1, j)
        elif value > value_above:
            count_hash[0] += 1
            hash_subset.insert(0, i)
            find_hash_subset(i - 1, j - weights[i])
        else:
            count_hash[0] += 1
            find_hash_subset(i - 1, j)


find_hash_subset(len(values) - 1, capacity)
print("(1c) Space-efficient Dynamic Programming Optimal subset:", '{' + ', '.join(map(str, hash_subset)) + '}')
print("(1c) Space-efficient Dynamic Programming Total Basic Ops:", count_hash[0])
print("(1c) Space-efficient Dynamic Programming Space Taken:" +  str(count_nodes[0 * k]) + "\n")


gs = greedySack(gd[0], gd[1], gd[2])
print("(2a) Greedy Approach Optimal value: " + str(gs[0]))
print("(2a) Greedy Approach Optimal subset: " + str(gs[1]))
print("(2a) Greedy Approach Total Basic Ops: " + str(greedySackOps(gd[0], gd[1], gd[2])) + "\n")

gh = heapSack(gd[0], gd[1], gd[2])
print("(2b) Heap-based Greedy Approach Optimal value: " + str(gh[0]))
print("(2b) Heap-based Greedy Approach Optimal subset: " + str(gh[1]))
print("(2b) Heap-based Greedy Approach Total Basic Ops: " + str(heapSackOps(gd[0], gd[1], gd[2])) + "\n")

