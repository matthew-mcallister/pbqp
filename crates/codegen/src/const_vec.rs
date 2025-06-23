use std::borrow::Borrow;
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct ConstVec<T: Copy, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: usize,
}

impl<T: Copy, const N: usize> ConstVec<T, N> {
    pub const fn new() -> Self {
        let data: MaybeUninit<[T; N]> = MaybeUninit::uninit();
        let data =
            unsafe { (&data as *const MaybeUninit<[T; N]> as *const [MaybeUninit<T>; N]).read() };
        Self { data, len: 0 }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn try_push(&mut self, value: T) -> Option<()> {
        if self.len >= N {
            return None;
        }
        self.data[self.len] = MaybeUninit::new(value);
        self.len += 1;
        Some(())
    }

    pub const fn push(&mut self, value: T) {
        self.try_push(value).expect("vector is full");
    }

    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let res = unsafe { *self.get_unchecked(self.len) };
        self.len -= 1;
        Some(res)
    }

    pub const unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { self.data[index].assume_init_ref() }
    }

    pub const fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        unsafe { Some(self.get_unchecked(index)) }
    }

    pub const fn last(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        unsafe { Some(self.get_unchecked(self.len() - 1)) }
    }

    pub const fn remove(&mut self, index: usize) -> T {
        unsafe {
            let val = self.data[index].assume_init_read();
            let count = self.len - index - 1;

            let ptr = self.data.as_mut_ptr();
            let src_ptr = ptr.add(index + 1);
            let dest_ptr = ptr.add(index);
            std::ptr::copy(src_ptr, dest_ptr, count);

            self.len -= 1;
            val
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        (0..self.len).map(|i| unsafe { self.get_unchecked(i) })
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        (0..self.len).map(move |i| unsafe { *self.get_unchecked(i) })
    }

    pub const fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    pub fn retain(&mut self, pred: impl Fn(&T) -> bool) {
        let mut i = 0;
        let mut j = 0;
        while i < self.len {
            let elem = unsafe { self.data[i].assume_init_ref() };
            if pred(elem) {
                self.data[j] = self.data[i];
                j += 1;
            }
            i += 1;
        }
        self.len = j;
    }
}

impl<T: Copy, const N: usize> Default for ConstVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> Clone for ConstVec<T, N> {
    fn clone(&self) -> Self {
        let mut new = ConstVec::new();
        for &elem in self.iter() {
            new.push(elem);
        }
        new
    }
}

impl<T: Copy, const N: usize> Copy for ConstVec<T, N> {}

impl<T: Copy, const N: usize> std::ops::Deref for ConstVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy, const N: usize> std::ops::DerefMut for ConstVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[macro_export]
macro_rules! const_vec {
    ($($item:expr),*$(,)?) => {
        const {
            let mut v = ConstVec::new();
            $(v.push($item);)*
            v
        }
    };
}

impl<T: Copy, const N: usize> Extend<T> for ConstVec<T, N> {
    fn extend<A: IntoIterator<Item = T>>(&mut self, iter: A) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Copy, const N: usize> FromIterator<T> for ConstVec<T, N> {
    fn from_iter<A: IntoIterator<Item = T>>(iter: A) -> Self {
        let mut vec = Self::new();
        vec.extend(iter);
        vec
    }
}

/// Fixed-size association list. Preserves insertion order. If `N < ~10` this
/// is generally faster than a hash map. Sadly cannot really be used in a const
/// context because `Eq` implementations cannot be const.
#[derive(Clone, Copy, Debug)]
pub struct ConstMap<K: Copy + Eq, V: Copy, const N: usize> {
    inner: ConstVec<(K, V), N>,
}

impl<K: Copy + Eq, V: Copy, const N: usize> Default for ConstMap<K, V, N> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<K: Copy + Eq, V: Copy, const N: usize> ConstMap<K, V, N> {
    pub const fn new() -> Self {
        Self {
            inner: ConstVec::new(),
        }
    }

    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    fn lookup<Q: Eq>(&self, key: &Q) -> Option<&(K, V)>
    where
        K: Borrow<Q>,
    {
        self.inner.iter().find(|(k, _)| k.borrow() == key)
    }

    fn index_of<Q: Eq>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
    {
        self.inner.iter().position(|(k, _)| k.borrow() == key)
    }

    fn lookup_mut<Q: Eq>(&mut self, key: &Q) -> Option<&mut (K, V)>
    where
        K: Borrow<Q>,
    {
        self.inner.iter_mut().find(|(k, _)| k.borrow() == key)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(entry) = self.lookup_mut(&key) {
            let v = entry.1;
            entry.1 = value;
            Some(v)
        } else {
            self.inner.push((key, value));
            None
        }
    }

    pub fn get<Q: Eq>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.lookup(key).map(|(_, v)| v)
    }

    pub fn get_mut<Q: Eq>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.lookup_mut(key).map(|(_, v)| v)
    }

    pub fn contains<Q: Eq>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.index_of(key).is_some()
    }

    pub fn entry<'a>(&'a mut self, key: K) -> ConstMapEntry<'a, K, V, N> {
        ConstMapEntry {
            index: self.index_of(&key),
            key,
            map: self,
        }
    }

    pub fn remove<Q: Eq>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        let i = self.index_of(key)?;
        let (_, v) = self.inner.remove(i);
        Some(v)
    }

    pub fn retain(&mut self, pred: impl Fn(&(K, V)) -> bool) {
        self.inner.retain(pred);
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)> + 'a {
        self.inner.iter().map(|(k, v)| (k, v))
    }

    pub fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K> + 'a {
        self.inner.iter().map(|(k, _)| k)
    }

    pub fn values<'a>(&'a self) -> impl Iterator<Item = &'a V> + 'a {
        self.inner.iter().map(|(_, v)| v)
    }
}

impl<K: Copy + Eq, V: Copy, const N: usize> FromIterator<(K, V)> for ConstMap<K, V, N> {
    fn from_iter<A: IntoIterator<Item = (K, V)>>(iter: A) -> Self {
        let mut out = Self::new();
        for (k, v) in iter {
            out.insert(k, v);
        }
        out
    }
}

#[derive(Debug)]
pub struct ConstMapEntry<'m, K: Copy + Eq, V: Copy, const N: usize> {
    map: &'m mut ConstMap<K, V, N>,
    key: K,
    index: Option<usize>,
}

impl<'m, K: Copy + Eq, V: Copy, const N: usize> ConstMapEntry<'m, K, V, N> {
    pub fn or_insert(self, value: V) -> &'m mut V {
        self.or_insert_with(|| value)
    }

    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'m mut V {
        let index = if let Some(index) = self.index {
            index
        } else {
            self.map.inner.push((self.key, f()));
            self.map.inner.len() - 1
        };
        unsafe { &mut self.map.inner.get_unchecked_mut(index).1 }
    }
}

impl<'m, K: Copy + Eq, V: Copy + Default, const N: usize> ConstMapEntry<'m, K, V, N> {
    pub fn or_default(self) -> &'m mut V {
        self.or_insert_with(Default::default)
    }
}
