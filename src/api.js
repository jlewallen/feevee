export function makeApiUrl(relative) {
  if (window.location.port >= 8080) {
    return "http://127.0.0.1:5000" + relative;
  }
  return window.location.origin + relative;
}

export function repeatedly(interval, callback) {
  callback().finally(function () {
    setTimeout(function () {
      return repeatedly(interval, callback);
    }, interval);
  });
}
