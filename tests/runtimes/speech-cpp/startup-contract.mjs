export async function checkSpeechStartup({ core, pointerBytes }) {
  if (core.pointerBytes !== pointerBytes || core.schema.runtime !== 'speech-cpp') {
    throw new Error('Wrong speech runtime or pointer width');
  }
  const apiVersionMajor = await core.api.speech_api_version_major();
  const apiVersionMinor = await core.api.speech_api_version_minor();
  if (apiVersionMajor !== core.constant('SPEECH_API_VERSION_MAJOR') || apiVersionMinor !== core.constant('SPEECH_API_VERSION_MINOR')) {
    throw new Error('Speech API version differs from the packaged schema');
  }
  const version = core.readUtf8(await core.api.speech_version());
  if (!/^\d+\.\d+\.\d+/u.test(version)) {
    throw new Error('Missing speech library version');
  }
  return { pointerBytes, apiVersionMajor, apiVersionMinor, version };
}
