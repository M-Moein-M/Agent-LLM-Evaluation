/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of RabbitCT.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <stdio.h>
#include <stdlib.h>

#include "rabbitProgress.h"

static const char **MRabbit;
static int MMax                   = 0;
static int MNextLine              = 1;
static int MRabbitLines           = 0;

static const char *Rabbit128[3]   = { " (\\_/)", "(='.'=)", "(\")_(\")" };

static const char *Rabbit256[9]   = { "   _     _     ",
    "   \\`\\ /`/   ",
    "    \\ V /     ",
    "    /. .\\     ",
    "   =\\ T /=    ",
    "    / ^ \\     ",
    "   /\\\\ //\\  ",
    " __\\ \" \" /__",
    "(____/^\\____) " };

static const char *Rabbit512[16]  = { "    / \\     / \\     ",
   "   {   }   {   }      ",
   "   {   {   }   }      ",
   "    \\   \\ /   /     ",
   "     \\   Y   /       ",
   "     .-\"`\"`\"-.     ",
   "   ,`         `.      ",
   "  /             \\    ",
   " /               \\   ",
   "{     ;\"\";,       } ",
   "{  /\";`'`,;       }  ",
   " \\{  ;`,'`;.     /   ",
   "  {  }`"
    "`  }   /}    ",
   "  {  }      {  //     ",
   "  {||}      {  /      ",
   "  `\"'       `\"'     " };

static const char *Rabbit1024[19] = { "        /|      __    ",
  "       / |   ,-~ /    ",
  "      Y :|  //  /     ",
  "      | jj /( .^      ",
  "      >-\"~\"-v\"     ",
  "     /       Y        ",
  "    jo  o    |        ",
  "   ( ~T~     j        ",
  "    >._-' _./         ",
  "   /   \"~\"  |       ",
  "  Y     _,  |         ",
  " /| ;-\"~ _  l        ",
  "/ l/ ,-\"~    \\      ",
  "\\//\\/      .- \\    ",
  " Y        /    Y*     ",
  " l       I     !      ",
  " ]\\      _\\    /\"\\",
  "(\" ~----( ~   Y.  )  ",
  "~~~~~~~~~~~~~~~~~~~   " };

void rabbitProgress_init(int maxValue, int rabbitSize)
{
  MMax = maxValue;

  if (rabbitSize >= 1024) {
    MRabbit      = Rabbit1024;
    MRabbitLines = 19;
  } else if (rabbitSize >= 512) {
    MRabbit      = Rabbit512;
    MRabbitLines = 16;
  } else if (rabbitSize >= 256) {
    MRabbit      = Rabbit256;
    MRabbitLines = 9;
  } else {
    MRabbit      = Rabbit128;
    MRabbitLines = 3;
  }
}

void rabbitProgress_progress(int pos)
{
  if ((float)MNextLine / (float)MRabbitLines <= (float)pos / (float)MMax) {
    printf(" %s\n", MRabbit[MNextLine - 1]);
    MNextLine++;
  }
}
