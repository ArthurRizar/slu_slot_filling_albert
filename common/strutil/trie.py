#coding:utf-8
###################################################
# File Name: trie.py
# Author: Meng Zhao
# mail: @
# Created Time: 2020年05月07日 星期四 14时31分18秒
#=============================================================
import codecs

import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerate_match(self, word, space="_", backward=False):  #space=''
        matched = []

        while len(word) > 0:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

if __name__ == '__main__':
    pass
