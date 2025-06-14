use std::mem::ManuallyDrop;

#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Id(u64);

impl Id {
    fn new(index: u32, generation: u32) -> Self {
        Id(((generation as u64) << 32) | (index as u64))
    }

    fn index(self) -> u32 {
        self.0 as u32
    }

    fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

impl std::fmt::Debug for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Id")
            .field(&self.index())
            .field(&self.generation())
            .finish()
    }
}

union Payload<T> {
    full: ManuallyDrop<T>,
    empty: u32,
}

/// Arena entry. If the generation counter is even, the entry is empty and
/// contains an index in the free list. If odd, the entry is full and contains
/// an element. The generation counter wraps around if it overflows a `u32`, so
/// it is possible for false positives to occur after a very long time.
struct Entry<T> {
    generation: u32,
    payload: Payload<T>,
}

impl<T> Entry<T> {
    fn is_empty(&self) -> bool {
        self.generation % 2 == 0
    }

    fn get(&self, generation: u32) -> Option<&T> {
        if !self.is_empty() && self.generation == generation {
            // safety: entry is full
            unsafe { Some(&*self.payload.full) }
        } else {
            None
        }
    }

    fn get_mut(&mut self, generation: u32) -> Option<&mut T> {
        if !self.is_empty() && self.generation == generation {
            // safety: entry is full
            unsafe { Some(&mut *self.payload.full) }
        } else {
            None
        }
    }

    // If full, remove and return the element, replacing it with a free list
    // index. Bumps the generation counter.
    fn take(&mut self, generation: u32, next: u32) -> Option<T> {
        if !self.is_empty() && self.generation == generation {
            // safety: entry is full
            let value = unsafe { ManuallyDrop::take(&mut self.payload.full) };
            self.generation = self.generation.wrapping_add(1);
            self.payload.empty = next;
            Some(value)
        } else {
            None
        }
    }

    // Replaces the current value with a new value. Panics if the entry is
    // already full. Bumps the generation counter.
    fn put(&mut self, value: T) -> u32 {
        if self.generation % 2 == 1 {
            panic!("Entry is already full");
        }
        self.generation = self.generation.wrapping_add(1);
        // safety: entry is full
        let next = unsafe { self.payload.empty };
        self.payload.full = ManuallyDrop::new(value);
        next
    }
}

impl<T> Drop for Entry<T> {
    fn drop(&mut self) {
        self.take(self.generation, 0);
    }
}

/// Generational arena.
pub struct Arena<T> {
    // Number of elements/occupied entries
    len: usize,
    // First entry in the free list. If free_head == entries.len(), then a new
    // entry will be pushed.
    free_head: u32,
    entries: Vec<Entry<T>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena {
            len: 0,
            free_head: 0,
            entries: Vec::new(),
        }
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn insert(&mut self, value: T) -> Id {
        let idx = self.free_head as usize;
        if idx == self.entries.len() {
            // No free slots, push new entry
            self.entries.push(Entry {
                generation: 1,
                payload: Payload {
                    full: ManuallyDrop::new(value),
                },
            });
            self.len += 1;
            self.free_head = self.entries.len() as u32;
            Id::new(idx as u32, 1)
        } else {
            let entry = &mut self.entries[idx];
            self.free_head = entry.put(value);
            self.len += 1;
            Id::new(idx as u32, entry.generation)
        }
    }

    pub fn remove(&mut self, id: Id) -> Option<T> {
        let idx = id.index() as usize;
        if idx >= self.entries.len() {
            return None;
        }
        let entry = &mut self.entries[idx];
        let generation = id.generation();
        let res = entry.take(generation, self.free_head as u32);
        if res.is_some() {
            self.free_head = idx as u32;
            self.len -= 1;
        }
        res
    }

    pub fn get(&self, id: Id) -> Option<&T> {
        let idx = id.index() as usize;
        if idx >= self.entries.len() {
            return None;
        }
        let entry = &self.entries[idx];
        entry.get(id.generation())
    }

    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        let idx = id.index() as usize;
        if idx >= self.entries.len() {
            return None;
        }
        let entry = &mut self.entries[idx];
        entry.get_mut(id.generation())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_head = 0;
        self.len = 0;
    }

    /// Returns an iterator over the elements contained in the arena, in no
    /// particular order.
    pub fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.entries.iter().enumerate().filter_map(|(idx, entry)| {
            if entry.is_empty() {
                None
            } else {
                let id = Id(((entry.generation as u64) << 32) | (idx as u64));
                entry.get(entry.generation).map(|v| (id, v))
            }
        })
    }
}

impl<T> std::ops::Index<Id> for Arena<T> {
    type Output = T;

    fn index(&self, index: Id) -> &Self::Output {
        self.get(index).expect("invalid index into arena")
    }
}

impl<T> std::ops::IndexMut<Id> for Arena<T> {
    fn index_mut(&mut self, index: Id) -> &mut Self::Output {
        self.get_mut(index).expect("invalid index into arena")
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::{Arena, Id};

    #[test]
    fn test_insert_get_delete() {
        let mut arena: Arena<u32> = Default::default();

        assert_eq!(arena.len(), 0);

        // Insert elements
        let a = arena.insert(10);
        assert_eq!(arena.len(), 1);
        let b = arena.insert(20);
        assert_eq!(arena.len(), 2);
        let c = arena.insert(30);
        assert_eq!(arena.len(), 3);

        // Lookup inserted elements
        assert_eq!(arena.get(a), Some(&10));
        assert_eq!(arena[a], 10);
        assert_eq!(arena.get(b), Some(&20));
        assert_eq!(arena[b], 20);
        assert_eq!(arena.get(c), Some(&30));
        assert_eq!(arena[c], 30);

        // Mutate an element
        *arena.get_mut(b).unwrap() = 25;
        assert_eq!(arena.get(b), Some(&25));
        arena[b] = 35;
        assert_eq!(arena[b], 35);
        assert_eq!(arena.len(), 3);

        // Remove an element
        assert_eq!(arena.remove(b), Some(35));
        assert_eq!(arena.get(b), None);
        assert_eq!(arena.len(), 2);

        // Insert another element
        let d = arena.insert(40);
        assert_eq!(arena[d], 40);
        assert_eq!(arena.len(), 3);

        // Remove all
        assert_eq!(arena.remove(a), Some(10));
        assert_eq!(arena.remove(c), Some(30));
        assert_eq!(arena.remove(d), Some(40));
        assert_eq!(arena.len(), 0);

        // All lookups should now be None
        assert_eq!(arena.get(a), None);
        assert_eq!(arena.get(b), None);
        assert_eq!(arena.get(c), None);
        assert_eq!(arena.get(d), None);

        // Insert again after clearing
        let e = arena.insert(50);
        assert_eq!(arena[e], 50);
    }

    #[test]
    #[should_panic]
    fn test_invalid_index_panics() {
        let mut arena: Arena<u32> = Default::default();
        let id = arena.insert(1);
        arena.remove(id);
        arena[id];
    }

    #[test]
    fn test_clear() {
        let mut arena: Arena<u32> = Default::default();
        let a = arena.insert(1);
        let b = arena.insert(2);

        assert_eq!(arena[a], 1);
        assert_eq!(arena[b], 2);

        arena.clear();

        assert_eq!(arena.len(), 0);
        assert_eq!(arena.get(a), None);
        assert_eq!(arena.get(b), None);

        // Insert after clear
        let c = arena.insert(3);
        assert_eq!(arena[c], 3);
    }

    #[test]
    fn test_destructor() {
        let mut arena: Arena<Rc<()>> = Default::default();
        let rc = Rc::new(());
        let a = arena.insert(rc.clone());
        assert_eq!(Rc::strong_count(&rc), 2);
        arena.remove(a);
        assert_eq!(Rc::strong_count(&rc), 1);

        arena.insert(rc.clone());
        arena.insert(rc.clone());
        assert_eq!(Rc::strong_count(&rc), 3);
        arena.clear();
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn test_iteration() {
        let mut arena: Arena<u32> = Default::default();
        let a = arena.insert(10);
        let b = arena.insert(20);
        let c = arena.insert(30);

        // Remove one element
        arena.remove(b);

        // Insert another element
        let d = arena.insert(40);

        // Collect all (id, value) pairs from the iterator
        let mut items: Vec<(Id, u32)> = arena.iter().map(|(id, &v)| (id, v)).collect();
        items.sort_by_key(|&(id, _)| id);

        // Only a, c, d should be present
        let mut expected = vec![(a, 10), (c, 30), (d, 40)];
        expected.sort_by_key(|&(id, _)| id);
        assert_eq!(items, expected);
    }
}
