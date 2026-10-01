# settings-cache

`SettingsStore` 是读写配置的**唯一入口**。

- `SettingsSource` 是慢源：每次 `read` 都是一次 I/O（生产环境读文件）。`source.reads`
  是它的读次数计数，供断言使用。
- `Cache` 只影响读性能。**缓存命中时不得触碰慢源。**
- 不变量：任何一次 `set` 之后，后续 `get` 必须看到新值，也不得因缓存命中而回退到慢源。
- `App` 只通过 `SettingsStore` 读写配置，不直接触碰慢源或缓存。
