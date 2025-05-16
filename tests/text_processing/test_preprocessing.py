from dackar.text_processing.Preprocessing import Preprocessing

class TestPreprocessing:

  content = ("bullet_points:\n"
        "\n‣ item1\n⁃ item2\n⁌ item3\n⁍ item4\n∙ item5\n▪ item6\n● item7\n◦ item8\n"
        "=======================\n"
        "hyphenated_words:\n"
        "I see you shiver with antici- pation.\n"
        "I see you shiver with antici-   \npation.\n"
        "I see you shiver with antici- PATION.\n"
        "I see you shiver with antici- 1pation.\n"
        "I see you shiver with antici pation.\n"
        "I see you shiver with antici-pation.\n"
        "My phone number is 555- 1234.\n"
        "I got an A- on the test.\n"
        "=======================\n"
        "quotation_marks:\n"
        "These are ´funny single quotes´.\n"
        "These are ‘fancy single quotes’.\n"
        "These are “fancy double quotes”.\n"
        "=======================\n"
        "repeating_chars:\n"
        "**Hello**, world!!! I wonder....... How are *you* doing?!?! lololol\n"
        "=======================\n"
        "unicode:\n"
        "Well… That's a long story.\n"
        "=======================\n"
        "whitespace:\n"
        "Hello,  world!\n"
        "Hello,     world!\n"
        "Hello,\tworld!\n"
        "Hello,\t\t  world!\n"
        "Hello,\n\nworld!\n"
        "Hello,\r\nworld!\n"
        "Hello\uFEFF, world!\n"
        "Hello\u200B\u200B, world!\n"
        "=======================\n"
        "accents:\n"
        "El niño se asustó del pingüino -- qué miedo!\n"
        "Le garçon est très excité pour la forêt.\n"
        "=======================\n"
        "brackets:\n"
        "Hello, {name}!\n"
        "Hello, world (DeWilde et al., 2021, p. 42)!\n"
        "Hello, world (1)!\n"
        "Hello, world [1]!\n"
        "Hello, world (and whomever it may concern [not that it's any of my business])!\n"
        "Hello, world (and whomever it may concern (not that it's any of my business))!\n"
        "Hello, world (and whomever it may concern [not that it's any of my business])!\n"
        "Hello, world [1]!\n"
        "Hello, world [1]!\n"
        "=======================\n"
        "html_tags:\n"
        "Hello, <i>world!</i>\n"
        "<title>Hello, world!</title>\n"
        '<title class="foo">Hello, world!</title>\n'
        "=======================\n"
        "punctuation:\n"
        "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience.\n"
        "=======================\n"
        "currency_symbols:\n"
        "$1.00 equals 100¢.\n"
        "How much is ¥100 in £?\n"
        "My password is 123$abc฿.\n"
        "=======================\n"
        "emails:\n"
        "Reach out at username@example.com.\n"
        "Click here: mailto:username@example.com.\n"
        "=======================\n"
        "emoji:\n"
        "ugh, it's raining *again* ☔\n"
        "✌ tests are passing ✌\n"
        "=======================\n"
        "hashtags:\n"
        "like omg it's #ThrowbackThursday\n"
        "#TextacyIn4Words: \"but it's honest work\"\n"
        "wth twitter #ican'teven #why-even-try\n"
        "www.foo.com#fragment is not a hashtag\n"
        "=======================\n"
        "numbers:\n"
        "I owe $1,000.99 to 123 people for 2 +1 reasons.\n"
        "=======================\n"
        "phone_numbers:\n"
        "I can be reached at 555-123-4567 through next Friday.\n"
        "=======================\n"
        "urls:\n"
        "I learned everything I know from www.stackoverflow.com and http://wikipedia.org/ and Mom.\n"
        "=======================\n"
        "user_handles:\n"
        "@Real_Burton_DeWilde: definitely not a bot\n"
        "wth twitter @b.j.dewilde\n"
        "foo@bar.com is not a user handle\n"
        "=======================\n"
        "numerize:\n"
        "forty-two\n"
        "four hundred and sixty two\n"
        "one fifty\n"
        "twelve hundred\n"
        "twenty one thousand four hundred and seventy three\n"
        "one billion and one\n"
        "nine and three quarters\n"
      )

  def get_preprocessing(self, preprocessorList, preprocessorOptions):
    preprocessing = Preprocessing(preprocessorList=preprocessorList, preprocessorOptions=preprocessorOptions)
    return preprocessing

  def test_preprocessing(self):
    preprocessorList = ['bullet_points',
                    'hyphenated_words',
                    'quotation_marks',
                    'repeating_chars',
                    'unicode',
                    'whitespace',
                    'accents',
                    'brackets',
                    'html_tags',
                    'punctuation',
                    'currency_symbols',
                    'emails',
                    'emojis',
                    'hashtags',
                    'numbers',
                    'phone_numbers',
                    'urls',
                    'user_handles',
                    'numerize']
    preprocessorOptions = {'repeating_chars': {'chars': 'ol', 'maxn': 2},
                        'unicode': {'form': 'NFKC'},
                        'accents': {'fast': False},
                        'brackets': {'only': 'square'},
                        'punctuation': {'only': '\''}}
    preprocessing = self.get_preprocessing(preprocessorList, preprocessorOptions)
    updated = preprocessing(self.content)
    print(updated)
    assert updated == """bullet_points:
  item1
  item2
  item3
  item4
  item5
  item6
  item7
  item8
=======================
hyphenated_words:
I see you shiver with anticipation.
I see you shiver with anticipation.
I see you shiver with anticiPATION.
I see you shiver with antici  1pation.
I see you shiver with antici pation.
I see you shiver with antici pation.
My phone number is _NUMBER_  _NUMBER_.
I got an 1  on the test.
=======================
quotation_marks:
These are funny single quotes .
These are fancy single quotes .
These are "fancy double quotes".
=======================
repeating_chars:
**Hello**, world!!! I wonder....... How are *you* doing?!?! lolol
=======================
unicode:
Well... That s a long story.
=======================
whitespace:
Hello, world!
Hello, world!
Hello, world!
Hello, world!
Hello,
world!
Hello,
world!
Hello, world!
Hello, world!
=======================
accents:
El nino se asusto del pinguino -  que miedo!
Le garcon est tres excite pour la foret.
=======================
brackets:
Hello, {name}!
Hello, world (DeWilde et al., _NUMBER_, p. _NUMBER_)!
Hello, world (_NUMBER_)!
Hello, world !
Hello, world (and whomever it may concern )!
Hello, world (and whomever it may concern (not that it s any of my business))!
Hello, world (and whomever it may concern )!
Hello, world !
Hello, world !
=======================
html_tags:
Hello, world!
Hello, world!
Hello, world!
=======================
punctuation:
I can t. No, I won t! It s a matter of "principle"; of -  what s the word? -  conscience.
=======================
currency_symbols:
_CUR_1.00 equals 100_CUR_.
How much is _CUR_100 in _CUR_?
My password is 123_CUR_abc_CUR_.
=======================
emails:
Reach out at username at example.com.
Click here: mailto:username at example.com.
=======================
emoji:
ugh, it s raining *again* _EMOJI_
_EMOJI_ tests are passing _EMOJI_
=======================
hashtags:
like omg it s _TAG_
_TAG_: "but it s honest work"
wth twitter _TAG_ teven _TAG_ even try
_URL_#fragment is not a hashtag
=======================
numbers:
I owe _CUR_1,000.99 to _NUMBER_ people for _NUMBER_ _NUMBER_ reasons.
=======================
phone_numbers:
I can be reached at _NUMBER_ _NUMBER_ _NUMBER_ through next Friday.
=======================
urls:
I learned everything I know from _URL_ and _URL_ and Mom.
=======================
user_handles:
 at Real_Burton_DeWilde: definitely not a bot
wth twitter at b.j.dewilde
foo at bar.com is not a user handle
=======================
numerize:
42
462
150
1200
21473
1000000001
9.75"""

